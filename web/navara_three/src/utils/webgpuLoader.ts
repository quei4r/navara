/**
 * Central dynamic loader for the WebGPU node system (three/webgpu, three/tsl,
 * and the TSL display helpers used by the post chain).
 *
 * The main entry statically pulls in the tile, picking, and orchestrator
 * modules; keeping these imports dynamic is what lets WebGL-only bundles
 * tree-shake the entire node system out. ThreeView kicks off loadWebGPU() in
 * its constructor on the WebGPU backend and awaits it in init(), before any
 * tile, pick, or post-processing code runs — so all synchronous consumers can
 * rely on getWebGPU().
 */
export type WebGPUModules = {
  webgpu: typeof import("three/webgpu");
  tsl: typeof import("three/tsl");
  bloom: typeof import("three/addons/tsl/display/BloomNode.js");
  fxaa: typeof import("three/addons/tsl/display/FXAANode.js");
};

let loaded: WebGPUModules | null = null;
let pending: Promise<WebGPUModules> | null = null;

/** Load the WebGPU node modules (idempotent, cached). */
export function loadWebGPU(): Promise<WebGPUModules> {
  if (loaded) return Promise.resolve(loaded);
  pending ??= Promise.all([
    import("three/webgpu"),
    import("three/tsl"),
    import("three/addons/tsl/display/BloomNode.js"),
    import("three/addons/tsl/display/FXAANode.js"),
  ]).then(([webgpu, tsl, bloom, fxaa]) => {
    loaded = { webgpu, tsl, bloom, fxaa };
    return loaded;
  });
  return pending;
}

/** True once loadWebGPU() has resolved. */
export function isWebGPULoaded(): boolean {
  return loaded !== null;
}

/**
 * Synchronous access to the loaded modules. Throws while loadWebGPU() has not
 * resolved yet — ThreeView.init() guarantees it has on the WebGPU backend.
 */
export function getWebGPU(): WebGPUModules {
  if (!loaded) {
    throw new Error(
      "WebGPU node modules not loaded: loadWebGPU() must resolve first " +
        "(ThreeView.init() does this on the WebGPU backend).",
    );
  }
  return loaded;
}
