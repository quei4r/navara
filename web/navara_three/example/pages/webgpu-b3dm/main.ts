/**
 * TEMP DEBUG PAGE (not committed): LACS Xiamen b3dm 3D-Tiles on the WebGPU
 * forward path. `?webgl` forces the WebGL2 backend, no flag = WebGPU.
 * `?classic` uses the stock WebGLRenderer path for upstream comparison.
 */
import ThreeView from "@navaramap/three";
import type { WebGLRenderer } from "three";
import { WebGPURenderer } from "three/webgpu";

const TILESET = "http://localhost:8787/city3d/xiamen/tileset.json";

const bootstrap = async () => {
  const canvas = document.createElement("canvas");
  canvas.id = "navara-canvas";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  const root = document.createElement("div");
  root.style.width = "100vw";
  root.style.height = "100vh";
  root.appendChild(canvas);
  document.body.appendChild(root);

  const query = new URLSearchParams(window.location.search);
  let renderer: WebGPURenderer | undefined;
  if (!query.has("classic")) {
    renderer = new WebGPURenderer({
      canvas,
      antialias: true,
      stencil: true,
      logarithmicDepthBuffer: true,
      forceWebGL: query.has("webgl"),
    });
    await renderer.init();
  }

  const view = new ThreeView({
    canvas,
    renderer: renderer as unknown as WebGLRenderer,
    animation: true,
  });
  await view.init();

  // Noon sun over Xiamen.
  const noon = new Date();
  noon.setUTCHours(4, 0, 0, 0);
  view.atmosphere.date = noon;

  const src = view.addSource({ type: "3d-tiles", url: TILESET });
  view.addLayer({
    type: "3d-tiles",
    source: src,
    model: { lit: query.has("lit") },
  });

  view.setCamera({
    lng: 118.075,
    lat: 24.457,
    height: 1600,
    heading: 0,
    pitch: -45,
    roll: 0,
  });

  (window as unknown as Record<string, unknown>).__b3dmDebug = {
    view,
    renderer,
  };
};

bootstrap().catch((e) => {
  console.error(e);
  const div = document.createElement("div");
  div.style.cssText =
    "position:fixed;inset:0;color:#f66;font:14px monospace;white-space:pre-wrap;padding:2em;";
  div.textContent = String(e?.stack ?? e);
  document.body.appendChild(div);
});
