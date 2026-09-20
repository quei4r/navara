import ThreeView from "@navaramap/three";
import type { WebGLRenderer } from "three";
import { WebGPURenderer } from "three/webgpu";

import { run, type CustomDescriptions } from "./run";

const bootstrap = async () => {
  const canvas = document.createElement("canvas");
  canvas.id = "navara-canvas";
  canvas.style.width = "100%";
  canvas.style.height = "100%";

  const root = document.createElement("div");
  root.id = "navara-root";
  root.style.width = "100vw";
  root.style.height = "100vh";
  root.appendChild(canvas);
  document.body.appendChild(root);

  const query = new URLSearchParams(window.location.search);

  // The renderer is created (and initialized) before the view: passing it via
  // `renderer` switches ThreeView to the experimental WebGPU forward path.
  // `?webgl` forces the WebGPURenderer's WebGL2 backend (same node pipeline,
  // no WebGPU device) for debugging. `?classic` uses the stock WebGLRenderer
  // path (A/B reference).
  let renderer: WebGPURenderer | undefined;
  if (!query.has("classic")) {
    renderer = new WebGPURenderer({
      canvas,
      antialias: true,
      stencil: true,
      logarithmicDepthBuffer: !query.has("nologdepth"),
      forceWebGL: query.has("webgl"),
    });
    await renderer.init();
  }

  // GPU device diagnostics (the jam compute passes are the heaviest GPU load
  // on this page — log device loss verbatim for field reports).
  const device = (
    renderer as unknown as { backend: { device: GPUDevice } } | undefined
  )?.backend?.device;
  if (device) {
    device.onuncapturederror = (e: GPUUncapturedErrorEvent) => {
      console.error(
        "[wgpu] uncaptured error:",
        (e.error as GPUError)?.message?.slice(0, 500),
      );
    };
    void device.lost.then((info) => {
      console.error("[wgpu] device lost:", info.reason, info.message);
    });
  }

  const view = new ThreeView<CustomDescriptions>({
    canvas,
    renderer: renderer as unknown as WebGLRenderer,
    animation: true,
  });
  await run(view, renderer as WebGPURenderer);
};

bootstrap().catch((e) => {
  console.error(e);
  const div = document.createElement("div");
  div.style.cssText =
    "position:fixed;inset:0;display:flex;align-items:center;justify-content:center;color:#f66;font:14px monospace;white-space:pre-wrap;padding:2em;";
  div.textContent = String(e?.stack ?? e);
  document.body.appendChild(div);
});
