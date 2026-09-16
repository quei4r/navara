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
  // no WebGPU device) for debugging.
  const renderer = new WebGPURenderer({
    canvas,
    antialias: true,
    stencil: true,
    logarithmicDepthBuffer: !query.has("nologdepth"),
    forceWebGL: query.has("webgl"),
  });
  await renderer.init();

  if (query.has("gpudebug")) {
    const device = (
      renderer as unknown as {
        backend: { device: GPUDevice };
      }
    ).backend?.device;
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
    } else {
      console.error("[wgpu] no device handle for debug");
    }
  }

  const view = new ThreeView<CustomDescriptions>({
    canvas,
    renderer: renderer as unknown as WebGLRenderer,
    animation: true,
    // WebGPU-path debug switches, exposed as URL flags for this demo page.
    webgpuDebug: {
      noEnv: query.has("noenv"),
      noPost: query.has("nopp"),
      rawPost: query.has("rawpp"),
      shadowOff: query.has("sh0"),
      tileBasic: query.has("tilebasic"),
    },
  });
  await run(view, renderer);
};

bootstrap().catch((e) => {
  console.error(e);
  const div = document.createElement("div");
  div.style.cssText =
    "position:fixed;inset:0;display:flex;align-items:center;justify-content:center;color:#f66;font:14px monospace;white-space:pre-wrap;padding:2em;";
  div.textContent = String(e?.stack ?? e);
  document.body.appendChild(div);
});
