/**
 * PictureInPicturePlugin — renders a secondary camera into a corner of the
 * main canvas.
 *
 * The window re-renders the view's shared scene containers on every
 * `postRender` with an independent `PerspectiveCamera`: tile LOD/loading
 * stays driven by the main camera, the PiP camera only draws what is already
 * resident. Rendering is direct-to-canvas with `setViewport`/`setScissor`;
 * the window gets no post-processing (bloom/FXAA) and no picking.
 *
 * ## Usage
 *
 * ```ts
 * import ThreeView, { PictureInPicturePlugin } from "@navaramap/three";
 *
 * const view = new ThreeView({ renderer }); // WebGPURenderer forward path
 * const pip = new PictureInPicturePlugin({ corner: "bottom-right" });
 * view.addPlugin(pip);
 * await view.init();
 *
 * pip.setCamera({ lng: 139.7671, lat: 35.6812, height: 120, pitch: -20 });
 * pip.hide(); pip.show(); pip.dispose();
 * ```
 */
import { Plugin } from "@navaramap/core";
import { headingPitchRollToFixedFrame } from "@navaramap/three-api";
import {
  Color,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  OrthographicCamera,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  Vector2,
  Vector4,
} from "three";

import type { ViewContext } from "../../core";
import type ThreeView from "../../index";

type View = ThreeView;

/** Scenes a window can render, listed in forward-path render order. */
export type PictureInPictureSceneKey =
  "globe" | "mrt" | "draped" | "opaque" | "transparent";

export type PictureInPictureCorner =
  "top-left" | "top-right" | "bottom-left" | "bottom-right";

/**
 * Window dock: a canvas corner (uses `margin`), or an explicit CSS-px offset
 * from the map canvas top-left — for aligning the window with an arbitrary
 * DOM panel instead of a corner.
 */
export type PictureInPicturePosition =
  | PictureInPictureCorner
  | { left: number; top: number };

/** Options for {@link PictureInPicturePlugin}. */
export type PictureInPicturePluginOptions = {
  /** Window width in CSS px. Defaults to `scale` × canvas width. */
  width?: number;
  /** Window height in CSS px. Defaults to width / `aspect`. */
  height?: number;
  /** Fraction of the canvas width used when `width` is omitted. Default 0.25. */
  scale?: number;
  /** Window aspect (width / height) used when `height` is omitted. Default 16/9. */
  aspect?: number;
  /** Canvas corner the window docks to. Default `"bottom-right"`. */
  corner?: PictureInPictureCorner;
  /** Gap between the window and the canvas edge, in CSS px. Default 12. */
  margin?: number;
  /** Initial visibility. Default `true`. */
  visible?: boolean;
  /**
   * Scenes rendered into the window, in render order. Defaults to the five
   * forward-path scenes (`globe` → `transparent`).
   */
  scenes?: PictureInPictureSceneKey[];
  /** Vertical field of view in degrees. Default 60. */
  fov?: number;
  /** Near plane in metres. Default 1. */
  near?: number;
  /** Far plane in metres. Default 12 000 000 (whole globe from ground level). */
  far?: number;
};

/** Geographic placement for the PiP camera. Angles in degrees. */
export type PictureInPictureCameraOptions = {
  /** Longitude in degrees. */
  lng: number;
  /** Latitude in degrees. */
  lat: number;
  /** Metres above the WGS84 ellipsoid. */
  height: number;
  /** Compass bearing the camera faces, degrees clockwise from north. */
  heading?: number;
  /** Degrees, nose up positive. */
  pitch?: number;
  /** Degrees, right wing down positive. */
  roll?: number;
  /** Vertical field of view in degrees. */
  fov?: number;
  /** Near plane in metres. */
  near?: number;
  /** Far plane in metres. */
  far?: number;
};

const DEFAULT_SCENES: PictureInPictureSceneKey[] = [
  "globe",
  "mrt",
  "draped",
  "opaque",
  "transparent",
];

// `headingPitchRollToFixedFrame` builds a frame whose +Z faces the heading
// (glTF convention); a camera looks down its -Z, so flip about Y.
const CAMERA_FLIP = new Matrix4().makeRotationY(Math.PI);

/**
 * Renders a secondary viewpoint into a docked corner window of the main
 * canvas. Add with `view.addPlugin(new PictureInPicturePlugin(...))` before
 * `view.init()`.
 */
export class PictureInPicturePlugin extends Plugin<View, ViewContext> {
  private view?: View;
  private ctx?: ViewContext;

  private readonly camera: PerspectiveCamera;
  private readonly scenes: PictureInPictureSceneKey[];
  private corner: PictureInPictureCorner;
  private margin: number;
  /** Explicit CSS-px offset from the canvas top-left; wins over `corner`. */
  private explicit?: { left: number; top: number };
  private width?: number;
  private height?: number;
  private scale: number;
  private aspect: number;
  private visible: boolean;

  // Background repaint quad: fills the window with the renderer clear color.
  // Depth is reset by an explicit depth clear (safe at postRender: the main
  // render has finished and nothing draws after us until the next frame).
  private readonly bgScene = new Scene();
  private readonly bgCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly bgGeometry = new PlaneGeometry(2, 2);
  private readonly bgMaterial = new MeshBasicMaterial({
    depthTest: false,
    depthWrite: false,
  });
  private readonly clearColor = new Color();

  private overlay?: HTMLDivElement;
  private resizeObserver?: ResizeObserver;

  private readonly canvasSize = new Vector2();
  private readonly prevViewport = new Vector4();
  private readonly prevScissor = new Vector4();

  private readonly boundPostRender: () => void;

  constructor(options: PictureInPicturePluginOptions = {}) {
    super();
    this.scenes = options.scenes ?? DEFAULT_SCENES;
    this.corner = options.corner ?? "bottom-right";
    this.margin = options.margin ?? 12;
    this.width = options.width;
    this.height = options.height;
    this.scale = options.scale ?? 0.25;
    this.aspect = options.aspect ?? 16 / 9;
    this.visible = options.visible ?? true;
    this.camera = new PerspectiveCamera(
      options.fov ?? 60,
      this.aspect,
      options.near ?? 1,
      options.far ?? 12_000_000,
    );
    const bgMesh = new Mesh(this.bgGeometry, this.bgMaterial);
    bgMesh.frustumCulled = false;
    bgMesh.position.z = -1;
    this.bgScene.add(bgMesh);
    this.boundPostRender = () => this.renderWindow();
  }

  async init(view: View, ctx: ViewContext): Promise<void> {
    this.view = view;
    this.ctx = ctx;
    view.on("postRender", this.boundPostRender);
    this.ensureOverlay();
  }

  /** The PiP camera. Never the view's main camera. */
  getCamera(): PerspectiveCamera {
    return this.camera;
  }

  /** Place the PiP camera geodetically (WGS84 lng/lat + ellipsoid height). */
  setCamera(options: PictureInPictureCameraOptions): void {
    const frame = headingPitchRollToFixedFrame({
      lng: options.lng,
      lat: options.lat,
      height: options.height,
      heading: options.heading,
      pitch: options.pitch,
      roll: options.roll,
    }).multiply(CAMERA_FLIP);
    frame.decompose(
      this.camera.position,
      this.camera.quaternion,
      this.camera.scale,
    );
    if (options.fov !== undefined) this.camera.fov = options.fov;
    if (options.near !== undefined) this.camera.near = options.near;
    if (options.far !== undefined) this.camera.far = options.far;
    this.camera.updateProjectionMatrix();
    this.view?.forceUpdate();
  }

  /** Set an explicit window size in CSS px; `undefined` restores deriving. */
  setSize(width?: number, height?: number): void {
    this.width = width;
    this.height = height;
    this.layoutOverlay();
    this.view?.forceUpdate();
  }

  /**
   * Dock the window: a canvas corner (optionally with a new margin), or an
   * explicit `{left, top}` CSS-px offset from the map canvas top-left.
   */
  setPosition(position: PictureInPicturePosition, margin?: number): void {
    if (typeof position === "string") {
      this.corner = position;
      this.explicit = undefined;
      if (margin !== undefined) this.margin = margin;
    } else {
      this.explicit = { left: position.left, top: position.top };
    }
    this.layoutOverlay();
    this.view?.forceUpdate();
  }

  show(): void {
    this.visible = true;
    if (this.overlay) this.overlay.hidden = false;
    this.view?.forceUpdate();
  }

  hide(): void {
    this.visible = false;
    if (this.overlay) this.overlay.hidden = true;
    this.view?.forceUpdate();
  }

  /** Detach listeners, the overlay, and the background-quad resources. */
  dispose(): void {
    this.view?.off("postRender", this.boundPostRender);
    this.resizeObserver?.disconnect();
    this.overlay?.remove();
    this.overlay = undefined;
    this.bgGeometry.dispose();
    this.bgMaterial.dispose();
    this.view = undefined;
    this.ctx = undefined;
  }

  /** Window rect in logical (CSS) px, or `undefined` when too small to draw. */
  private windowRect(
    renderer: ReturnType<ViewContext["getRenderer"]>,
  ): { x: number; y: number; w: number; h: number } | undefined {
    const size = this.canvasSize;
    const w = Math.floor(this.width ?? size.x * this.scale);
    const h = Math.floor(this.height ?? w / this.aspect);
    if (w < 8 || h < 8) return undefined;
    const x = this.explicit
      ? this.explicit.left
      : this.corner === "top-right" || this.corner === "bottom-right"
        ? size.x - this.margin - w
        : this.margin;
    const top = this.explicit
      ? this.explicit.top
      : this.corner === "top-left" || this.corner === "top-right"
        ? this.margin
        : size.y - this.margin - h;
    // WebGPURenderer measures viewport/scissor y from the attachment's top
    // edge on both its backends (the WebGL fallback flips it back); the
    // classic WebGLRenderer measures from the bottom.
    const topOrigin =
      (renderer as unknown as { isWebGPURenderer?: boolean })
        .isWebGPURenderer === true;
    return { x, y: topOrigin ? top : size.y - top - h, w, h };
  }

  private renderWindow(): void {
    const ctx = this.ctx;
    if (!ctx || !this.visible) return;
    const renderer = ctx.getRenderer();
    renderer.getSize(this.canvasSize);
    const rect = this.windowRect(renderer);
    if (!rect) return;

    const aspect = rect.w / rect.h;
    if (this.camera.aspect !== aspect) {
      this.camera.aspect = aspect;
      this.camera.updateProjectionMatrix();
    }

    const prevTarget = renderer.getRenderTarget();
    renderer.getViewport(this.prevViewport);
    renderer.getScissor(this.prevScissor);
    const prevScissorTest = renderer.getScissorTest();

    renderer.setRenderTarget(null);
    renderer.setViewport(rect.x, rect.y, rect.w, rect.h);
    renderer.setScissor(rect.x, rect.y, rect.w, rect.h);
    renderer.setScissorTest(true);

    renderer.clear(false, true, false);
    renderer.getClearColor(this.clearColor);
    this.bgMaterial.color.copy(this.clearColor);
    renderer.render(this.bgScene, this.bgCamera);

    // Light the scenes the same way the orchestrator does; the groups are
    // re-parented scene to scene (a Group has a single parent), which is
    // safe because each scene has already been rendered by the main pass.
    const lights = ctx.getRenderLights();
    for (const key of this.scenes) {
      const scene = ctx.scenes[key];
      for (const light of lights) {
        if (light.children.length > 0) scene.add(light);
      }
      renderer.render(scene, this.camera);
    }
    for (const key of this.scenes) {
      const scene = ctx.scenes[key];
      for (const light of lights) scene.remove(light);
    }

    renderer.setViewport(this.prevViewport);
    renderer.setScissor(this.prevScissor);
    renderer.setScissorTest(prevScissorTest);
    renderer.setRenderTarget(prevTarget);
  }

  /** Thin frame marking the window bounds; carries no interaction. */
  private ensureOverlay(): void {
    if (typeof document === "undefined") return;
    const canvas = this.ctx?.getRenderer().domElement;
    const parent = canvas?.parentElement;
    if (!canvas || !parent) return;
    if (getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
    }
    const overlay = document.createElement("div");
    overlay.style.cssText =
      "position:absolute;pointer-events:none;box-sizing:border-box;" +
      "border:1px solid rgba(255,255,255,0.7);box-shadow:0 0 0 1px rgba(0,0,0,0.5);";
    overlay.hidden = !this.visible;
    parent.appendChild(overlay);
    this.overlay = overlay;
    this.resizeObserver = new ResizeObserver(() => this.layoutOverlay());
    this.resizeObserver.observe(canvas);
    this.layoutOverlay();
  }

  private layoutOverlay(): void {
    const overlay = this.overlay;
    const canvas = this.ctx?.getRenderer().domElement;
    if (!overlay || !canvas) return;
    const w = Math.floor(this.width ?? canvas.clientWidth * this.scale);
    const h = Math.floor(this.height ?? w / this.aspect);
    const left = this.explicit
      ? canvas.offsetLeft + this.explicit.left
      : this.corner === "top-right" || this.corner === "bottom-right"
        ? canvas.offsetLeft + canvas.clientWidth - this.margin - w
        : canvas.offsetLeft + this.margin;
    const top = this.explicit
      ? canvas.offsetTop + this.explicit.top
      : this.corner === "top-left" || this.corner === "top-right"
        ? canvas.offsetTop + this.margin
        : canvas.offsetTop + canvas.clientHeight - this.margin - h;
    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${w}px`;
    overlay.style.height = `${h}px`;
  }
}
