import { NamedIndexMap } from "@navaramap/core";
import { EffectComposer, Pass as PostProcessingPass } from "postprocessing";
import {
  AmbientLight,
  DirectionalLight,
  FloatType,
  HalfFloatType,
  HemisphereLight,
  Mesh,
  OrthographicCamera,
  PlaneGeometry,
  PointLight,
  Scene,
  SpotLight,
  Vector2,
  WebGLRenderer,
  Group,
} from "three";
import type { Object3D, Texture, Vector3 } from "three";
import type { PostProcessing, RenderTarget } from "three/webgpu";

import { estimateFixedGpuBytes } from "../utils/fixedGpuFootprint";
import { getWebGPU, isWebGPULoaded } from "../utils/webgpuLoader";

export type RenderPassOrchestratorOptions = {
  halfFloat?: boolean;
  multisampling?: number;
  /** Debug switches for the experimental WebGPU forward path. */
  debugFlags?: WebGpuDebugFlags;
};

/**
 * Debug switches for the experimental WebGPU forward path. All default to
 * false (the production behavior). Exposed to applications as
 * `Options.webgpuDebug`.
 */
export type WebGpuDebugFlags = {
  /** Skip the sky/sun/shadow environment entirely. */
  noEnv?: boolean;
  /** Render scenes straight to the canvas, skipping the TSL post chain. */
  noPost?: boolean;
  /** Skip FXAA in the post chain (raw scene + bloom output). */
  rawPost?: boolean;
  /** Disable shadow mapping (castShadow + shadow map updates). */
  shadowOff?: boolean;
  /** Force tiles to the plain unlit single-texture material (A/B regression). */
  tileBasic?: boolean;
};

export type NamedPass = {
  name: string;
  pass: PostProcessingPass;
};

type LightProxy = { proxy: Object3D; sync: () => void };

/**
 * Build a per-frame-synced stand-in for an app light from `scenes.light`.
 *
 * The WebGL path nests the whole light scene into each rendered scene
 * (CustomRenderPass._renderWithLight). On WebGPU that works for plain lights
 * but not for CascadedDirectionalLights: its per-cascade clones rely on the
 * GLSL chunk swap to keep only the main cascade's contribution, which TSL
 * materials never run — nesting the group would stack the extra white
 * clones. So a CSM group collapses to a single directional proxy driven by
 * mainLight + direction, exactly what the classic injected shader resolves
 * to. Proxies never cast shadows: the WebGPUEnvironment's native sun owns
 * shadow mapping on this path. Only top-level lights are mirrored.
 */
function makeLightProxy(source: Object3D): LightProxy | null {
  const csm = source as {
    mainLight?: DirectionalLight;
    direction?: Vector3;
    cascadedLights?: DirectionalLight[];
  };
  if (
    csm.mainLight?.isDirectionalLight &&
    csm.direction &&
    Array.isArray(csm.cascadedLights)
  ) {
    const main = csm.mainLight;
    const direction = csm.direction;
    const proxy = new DirectionalLight();
    proxy.castShadow = false;
    const sync = () => {
      proxy.color.copy(main.color);
      proxy.intensity = main.intensity;
      proxy.visible = source.visible && main.visible;
      // Parallel light: only the position→target vector matters.
      proxy.position.copy(direction).multiplyScalar(-1e7);
      proxy.target.position.set(0, 0, 0);
    };
    sync();
    return { proxy, sync };
  }
  const src = source as Partial<
    AmbientLight & DirectionalLight & HemisphereLight & PointLight & SpotLight
  >;
  if (src.isAmbientLight) {
    const s = source as AmbientLight;
    const proxy = new AmbientLight();
    const sync = () => {
      proxy.color.copy(s.color);
      proxy.intensity = s.intensity;
      proxy.visible = s.visible;
    };
    sync();
    return { proxy, sync };
  }
  if (src.isHemisphereLight) {
    const s = source as HemisphereLight;
    const proxy = new HemisphereLight();
    const sync = () => {
      proxy.color.copy(s.color);
      proxy.groundColor.copy(s.groundColor);
      proxy.intensity = s.intensity;
      proxy.visible = s.visible;
    };
    sync();
    return { proxy, sync };
  }
  if (src.isSpotLight) {
    const s = source as SpotLight;
    const proxy = new SpotLight();
    proxy.castShadow = false;
    const sync = () => {
      proxy.color.copy(s.color);
      proxy.intensity = s.intensity;
      proxy.distance = s.distance;
      proxy.decay = s.decay;
      proxy.angle = s.angle;
      proxy.penumbra = s.penumbra;
      proxy.visible = s.visible;
      proxy.position.copy(s.position);
      proxy.target.position.copy(s.target.position);
    };
    sync();
    return { proxy, sync };
  }
  if (src.isPointLight) {
    const s = source as PointLight;
    const proxy = new PointLight();
    proxy.castShadow = false;
    const sync = () => {
      proxy.color.copy(s.color);
      proxy.intensity = s.intensity;
      proxy.distance = s.distance;
      proxy.decay = s.decay;
      proxy.visible = s.visible;
      proxy.position.copy(s.position);
    };
    sync();
    return { proxy, sync };
  }
  if (src.isDirectionalLight) {
    const s = source as DirectionalLight;
    const proxy = new DirectionalLight();
    proxy.castShadow = false;
    const sync = () => {
      proxy.color.copy(s.color);
      proxy.intensity = s.intensity;
      proxy.visible = s.visible;
      proxy.position.copy(s.position);
      proxy.target.position.copy(s.target.position);
    };
    sync();
    return { proxy, sync };
  }
  return null;
}

// Implementation policy:
// - Here we only manage Pass with an ordered Map mechanism.
// - When a Layer inserts with insertBefore/After, it inserts in the appropriate place.
// - If we render in order from the top, we can render while preserving dependencies.
// - As a preliminary step to the Layer mechanism, we want to be able to just freely add Pass from the ThreeView to the RenderPassOrchestrator.
// - Use NamedIndexMap for Pass management.

/**
 * Orchestrate rendering passes with ordered management and flexible insertion.
 *
 * The WebGPU backend (`backend: "webgpu"`) is an experimental forward path: no
 * EffectComposer is created and `render()` draws the pipeline scenes directly
 * to the canvas in dependency order (globe → mrt → draped → opaque →
 * transparent). Pass registration is kept for ordering bookkeeping, but
 * postprocessing-library passes are not executed on this backend.
 */
export class RenderPassOrchestrator {
  lights = new Group();
  scenes = {
    mrt: new Scene(),
    globe: new Scene(),
    draped: new Scene(),
    opaque: new Scene(),
    transparent: new Scene(),
    skyEnvMap: new Scene(),
  };
  effectComposer: EffectComposer | undefined;

  /**
   * Mirrors of the app-configured lights in `scenes.light`, attached to each
   * rendered scene on the WebGPU path (see makeLightProxy). Rebuilt when the
   * light scene's children change; synced every frame.
   */
  private readonly lightMirror = new Group();
  private lightProxies: LightProxy[] = [];
  private lightMirrorDirty = true;
  private lightSceneListened?: Scene;

  /**
   * The active renderer. Typed as WebGLRenderer for API compatibility; on the
   * WebGPU backend this holds a (duck-compatible) WebGPURenderer instance.
   */
  renderer: WebGLRenderer;

  backend: "webgl" | "webgpu";

  private mainCamera?: import("three").Camera;

  /**
   * Invoked whenever the pass list changes (add/insert/remove/clear), so the
   * owner can re-report the fixed GPU footprint to the memory ledger.
   */
  onPassesChanged?: () => void;

  private passMap = new NamedIndexMap<NamedPass>();

  /** Debug switches for the WebGPU forward path (never set in production). */
  readonly debugFlags: WebGpuDebugFlags;

  // --- WebGPU scene-depth sampling (zoom-to-cursor support) ---------------
  // TerrainPicker's GLSL depth-sample pass is WebGL-only; on the WebGPU
  // forward path the depth lives in the scene target's depth texture. GPU
  // readbacks are async, so the synchronous pick API (pickDepthPosition,
  // called per wheel event) consumes the most recent completed sample and
  // each request schedules a fresh capture on the next render.
  private depthSampleRequest: [number, number] | null = null; // CSS px
  private depthSampleValue: number | null = null;
  private depthSampleReadInFlight = false;
  private depthSample?: {
    target: RenderTarget;
    scene: Scene;
    camera: OrthographicCamera;
    uv: { value: Vector2 };
    texNode: { value: Texture };
  };

  /**
   * WebGPU only: returns the latest asynchronously-read scene depth (NDC
   * [0, 1]) at the most recently requested screen point, or null until the
   * first readback lands (and on the noPost debug path, where no scene
   * target exists). Schedules a fresh capture at (`x`, `y`) — CSS pixels —
   * for the next render.
   */
  requestDepthSample(x: number, y: number): number | null {
    this.depthSampleRequest = [x, y];
    return this.depthSampleValue;
  }

  private captureDepthSample(target: RenderTarget): void {
    const { webgpu, tsl } = getWebGPU();
    if (!this.depthSample) {
      const rt = new webgpu.RenderTarget(1, 1, {
        type: FloatType,
        depthBuffer: false,
        stencilBuffer: false,
      });
      const uv = tsl.uniform(new Vector2(0.5, 0.5));
      const depthNode = tsl.texture(new webgpu.DepthTexture(1, 1));
      const material = new webgpu.MeshBasicNodeMaterial({
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      });
      material.colorNode = tsl.vec4(depthNode.sample(uv).x, 0, 0, 1);
      const scene = new Scene();
      scene.add(new Mesh(new PlaneGeometry(2, 2), material));
      this.depthSample = {
        target: rt,
        scene,
        camera: new OrthographicCamera(-1, 1, 1, -1, 0, 1),
        uv: uv as unknown as { value: Vector2 },
        texNode: depthNode as unknown as { value: Texture },
      };
    }
    const request = this.depthSampleRequest;
    this.depthSampleRequest = null;
    const depthTexture = target.depthTexture;
    const sample = this.depthSample;
    if (!request || !depthTexture || !sample) return;

    // The depth texture node must follow target recreation (resize).
    sample.texNode.value = depthTexture;
    // Sample the exact texel center: WebGPU depth texture row 0 is NDC
    // y=+1 (screen top), so v = cy / height — no flip.
    const renderer = this.renderer;
    const size = renderer.getDrawingBufferSize(new Vector2());
    const pixelRatio = renderer.getPixelRatio();
    const cx =
      Math.max(0, Math.min(size.x - 1, Math.floor(request[0] * pixelRatio))) +
      0.5;
    const cy =
      Math.max(0, Math.min(size.y - 1, Math.floor(request[1] * pixelRatio))) +
      0.5;
    sample.uv.value.set(cx / size.x, cy / size.y);

    const prevTarget = renderer.getRenderTarget();
    renderer.setRenderTarget(
      sample.target as unknown as Parameters<WebGLRenderer["setRenderTarget"]>[0],
    );
    renderer.render(sample.scene, sample.camera);
    renderer.setRenderTarget(prevTarget);

    if (this.depthSampleReadInFlight) return;
    this.depthSampleReadInFlight = true;
    (
      renderer as unknown as {
        readRenderTargetPixelsAsync: (
          t: RenderTarget,
          x: number,
          y: number,
          w: number,
          h: number,
        ) => Promise<ArrayBufferView>;
      }
    )
      .readRenderTargetPixelsAsync(sample.target, 0, 0, 1, 1)
      .then((pixels) => {
        this.depthSampleValue = (pixels as Float32Array)[0];
        this.depthSampleReadInFlight = false;
      })
      .catch(() => {
        this.depthSampleReadInFlight = false;
      });
  }

  constructor(renderer: WebGLRenderer, options: RenderPassOrchestratorOptions) {
    this.renderer = renderer;
    this.debugFlags = options.debugFlags ?? {};
    this.backend = (renderer as { isWebGPURenderer?: boolean }).isWebGPURenderer
      ? "webgpu"
      : "webgl";
    if (this.backend === "webgl") {
      // Setup render pass
      this.effectComposer = new EffectComposer(renderer, {
        stencilBuffer: true,
        frameBufferType:
          (options.halfFloat ?? true) ? HalfFloatType : undefined,
        multisampling: options.multisampling,
      });
    }
  }

  setMainCamera(camera: import("three").Camera) {
    this.mainCamera = camera;
    this.effectComposer?.setMainCamera(camera);
  }

  /**
   * True when the app has configured lights of its own in `scenes.light`
   * (LightDesc adds them there). Drives WebGPUEnvironment's fallback rig.
   */
  get hasAppLights(): boolean {
    return ((this.scenes as { light?: Scene }).light?.children.length ?? 0) > 0;
  }

  /** Rebuild (on light-scene changes) and per-frame sync the light mirror. */
  private syncLightMirror(): void {
    const lightScene = (this.scenes as { light?: Scene }).light;
    if (lightScene && this.lightSceneListened !== lightScene) {
      this.lightSceneListened = lightScene;
      const dirty = () => {
        this.lightMirrorDirty = true;
      };
      lightScene.addEventListener("childadded", dirty);
      lightScene.addEventListener("childremoved", dirty);
      this.lightMirrorDirty = true;
    }
    if (this.lightMirrorDirty) {
      this.lightMirrorDirty = false;
      this.lightProxies = [];
      this.lightMirror.clear();
      for (const child of lightScene?.children ?? []) {
        const entry = makeLightProxy(child);
        if (!entry) continue;
        this.lightProxies.push(entry);
        this.lightMirror.add(entry.proxy);
        const target = (entry.proxy as DirectionalLight | SpotLight).target;
        if (target) this.lightMirror.add(target);
      }
    }
    for (const p of this.lightProxies) p.sync();
  }

  setSize(width: number, height: number) {
    this.effectComposer?.setSize(width, height);
  }

  render() {
    if (this.backend === "webgpu") {
      if (!this.mainCamera) {
        return;
      }
      const renderer = this.renderer;
      // Forward path with a TSL PostProcessing chain: the five scenes render
      // into an HDR target, then bloom → FXAA run as nodes and the output
      // transform applies the renderer's tone mapping + color space. Manual
      // clear control (autoClear* are disabled by ThreeView); the light group
      // attaches temporarily to whichever scene is being rendered, matching
      // CustomRenderPass's semantics on WebGL.
      const noPP =
        (this.debugFlags.noPost ?? false) ||
        // First frames can arrive before the dynamically imported node
        // system resolves; render straight to the canvas until then.
        !isWebGPULoaded();
      const target = noPP ? null : (this.ensureSceneTarget() ?? null);
      renderer.setRenderTarget(
        target as unknown as Parameters<WebGLRenderer["setRenderTarget"]>[0],
      );
      renderer.clear(true, true, true);
      const scenes = [
        this.scenes.globe,
        this.scenes.mrt,
        this.scenes.draped,
        this.scenes.opaque,
        this.scenes.transparent,
      ];
      this.syncLightMirror();
      const lightsAttached = new Set<Scene>();
      for (const scene of scenes) {
        if (
          this.lights.children.length > 0 &&
          !scene.children.includes(this.lights)
        ) {
          scene.add(this.lights);
          lightsAttached.add(scene);
        }
        if (
          this.lightMirror.children.length > 0 &&
          !scene.children.includes(this.lightMirror)
        ) {
          scene.add(this.lightMirror);
          lightsAttached.add(scene);
        }
        renderer.render(scene, this.mainCamera);
      }
      for (const scene of lightsAttached) {
        scene.remove(this.lights);
        scene.remove(this.lightMirror);
      }
      // Back to the canvas before the post chain — the PP quad samples the
      // scene target, so it must not render into it.
      if (this.depthSampleRequest && target?.depthTexture) {
        this.captureDepthSample(target);
      }
      renderer.setRenderTarget(null);
      if (!noPP) {
        this.scenePostProcessing?.render();
      }
      return;
    }
    this.effectComposer?.render();
  }

  private sceneTarget?: RenderTarget;
  private sceneTargetSize = new Vector2();
  private scenePostProcessing?: PostProcessing;

  /**
   * The HDR intermediate the five scenes render into, plus the bloom+FXAA
   * PostProcessing that composites it to the canvas.
   */
  private ensureSceneTarget(): RenderTarget | undefined {
    const renderer = this.renderer;
    const size = renderer.getDrawingBufferSize(this.sceneTargetSize);
    const width = Math.max(1, Math.floor(size.x || 1));
    const height = Math.max(1, Math.floor(size.y || 1));
    const current = this.sceneTarget;
    if (current && current.width === width && current.height === height) {
      return current;
    }
    current?.dispose();
    this.scenePostProcessing?.dispose();

    const { webgpu, tsl, bloom: bloomMod, fxaa: fxaaMod } = getWebGPU();
    const target = new webgpu.RenderTarget(width, height, {
      type: HalfFloatType,
    });
    // Attach a sampleable depth texture: pickDepthPosition's WebGPU path
    // reads it back (async, 1 texel) to anchor zoom-to-cursor.
    const depthTexture = new webgpu.DepthTexture(width, height);
    depthTexture.isRenderTargetTexture = true;
    target.depthTexture = depthTexture;
    this.sceneTarget = target;
    const sceneColor = tsl.texture(target.texture);
    const rawPP = this.debugFlags.rawPost ?? false;
    // bloom() yields the glow contribution only — compose over the scene.
    const bloomNode = bloomMod.bloom(sceneColor, 0.35, 0.4, 0.9);
    const withBloom = sceneColor.add(bloomNode);
    const output = rawPP ? sceneColor : fxaaMod.fxaa(withBloom);
    const postProcessing = new webgpu.PostProcessing(
      renderer as unknown as ConstructorParameters<typeof PostProcessing>[0],
    );
    postProcessing.outputNode = output;
    this.scenePostProcessing = postProcessing;
    return target;
  }

  /**
   * Add a named pass to the end of the pass list.
   */
  addPass(name: string, pass: PostProcessingPass): void {
    const namedPass: NamedPass = { name, pass };
    this.passMap.add(namedPass);
    this.effectComposer?.addPass(pass);
    this.onPassesChanged?.();
  }

  /**
   * Insert a pass before the specified target pass.
   */
  insertPassBefore(
    targetName: string,
    name: string,
    pass: PostProcessingPass,
  ): void {
    const namedPass: NamedPass = { name, pass };
    const targetIndex = this.passMap.insertBefore(targetName, namedPass);
    this.effectComposer?.addPass(pass, targetIndex);
    this.onPassesChanged?.();
  }

  /**
   * Insert a pass after the specified target pass.
   */
  insertPassAfter(
    targetName: string,
    name: string,
    pass: PostProcessingPass,
  ): void {
    const namedPass: NamedPass = { name, pass };
    const targetIndex = this.passMap.insertAfter(targetName, namedPass);
    this.effectComposer?.addPass(pass, targetIndex);
    this.onPassesChanged?.();
  }

  /**
   * Remove a pass by name.
   */
  removePass(name: string): void {
    const targetPass = this.passMap.list.find((p) => p.name === name);
    if (!targetPass) {
      throw new Error(`Pass not found: ${name}`);
    }

    this.effectComposer?.removePass(targetPass.pass);
    this.passMap.list = this.passMap.list.filter((p) => p.name !== name);
    this.rebuildIndexMap();
    this.onPassesChanged?.();
  }

  /**
   * Get a pass by name.
   */
  getPass(name: string): PostProcessingPass | undefined {
    return this.passMap.list.find((p) => p.name === name)?.pass;
  }

  /**
   * Get all pass names in order.
   */
  getPassNames(): string[] {
    return this.passMap.list.map((p) => p.name);
  }

  /**
   * Clear all passes.
   */
  clearPasses(): void {
    for (const namedPass of this.passMap.list) {
      this.effectComposer?.removePass(namedPass.pass);
    }
    this.passMap.list = [];
    this.passMap.indexMap = {};
    this.onPassesChanged?.();
  }

  /**
   * Estimates the resident GPU bytes of the fixed, screen-sized render-target
   * stack (composer buffers + every target reachable from the passes), for
   * reporting into the memory ledger's `fixed_gpu_bytes` term.
   */
  estimateFixedGpuBytes(): number {
    if (!this.effectComposer) {
      return 0;
    }
    return estimateFixedGpuBytes(
      this.effectComposer,
      this.passMap.list.map((p) => p.pass),
    );
  }

  /**
   * Dispose all GPU resources held by the effect composer.
   */
  dispose(): void {
    this.clearPasses();
    this.effectComposer?.dispose();
  }

  /**
   * Rebuild the index map after manual list modification.
   */
  private rebuildIndexMap(): void {
    this.passMap.indexMap = {};
    for (let i = 0; i < this.passMap.list.length; i++) {
      this.passMap.indexMap[this.passMap.list[i].name] = i;
    }
  }
}
