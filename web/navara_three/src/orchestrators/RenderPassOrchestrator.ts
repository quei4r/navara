import { NamedIndexMap } from "@navaramap/core";
import { EffectComposer, Pass as PostProcessingPass } from "postprocessing";
import { HalfFloatType, Scene, Vector2, WebGLRenderer, Group } from "three";
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
      const lightsAttached = new Set<Scene>();
      for (const scene of scenes) {
        if (
          this.lights.children.length > 0 &&
          !scene.children.includes(this.lights)
        ) {
          scene.add(this.lights);
          lightsAttached.add(scene);
        }
        renderer.render(scene, this.mainCamera);
      }
      for (const scene of lightsAttached) {
        scene.remove(this.lights);
      }
      // Back to the canvas before the post chain — the PP quad samples the
      // scene target, so it must not render into it.
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
