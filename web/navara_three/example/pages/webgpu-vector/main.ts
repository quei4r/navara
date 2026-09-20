/**
 * TEMP DEBUG PAGE (not committed): LACS GBA vector-tile extrusion layer on
 * the WebGPU forward path. `?classic` uses the stock WebGLRenderer path.
 */
import ThreeView, {
  Color,
  TERRARIUM_ELEVATION_DECODER,
} from "@navaramap/three";
import type { WebGLRenderer } from "three";
import { WebGPURenderer } from "three/webgpu";

import { TERRAIN_DATASETS } from "../../helpers/constants";

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

  // Noon sun.
  const noon = new Date();
  noon.setUTCHours(4, 0, 0, 0);
  view.atmosphere.date = noon;

  // ?terrain adds a DEM terrain layer (mapterhorn, same as jamfield).
  if (query.has("terrain")) {
    const dem = view.addSource({
      type: "raster-dem",
      url: TERRAIN_DATASETS.mapterhorn.url,
      maxZoom: 17,
      minZoom: 5,
      elevationDecoder: TERRARIUM_ELEVATION_DECODER(),
      tileSize: 512,
    });
    view.addLayer({ type: "terrain", source: dem, terrain: {} });
  }

  const src = query.has("novec")
    ? null
    : view.addSource({
        type: "vector-tile",
        url: "http://localhost:8787/tiles/{z}/{x}/{y}.mvt",
        maxZoom: 15,
      });
  const buildingLayer = src
    ? view.addLayer({
        type: "vector",
        source: src,
        sourceLayers: ["building"],
        polygon: {
          color: new Color().setStyle("#9fb3c8"),
          clampToGround: false,
          height: 0,
          extrudedHeight: 80,
        },
      })
    : null;
  buildingLayer?.on("featureUpdated", ({ evaluator }) => {
    evaluator.evaluate(({ properties }) => {
      const h = (properties?.["height"] as number) ?? 6;
      return { extrudedHeight: h > 0 ? h : 6 };
    });
  });

  view.setCamera({
    lng: 118.075,
    lat: 24.457,
    height: 1600,
    heading: 0,
    pitch: -45,
    roll: 0,
  });

  (window as unknown as Record<string, unknown>).__vecDebug = {
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
