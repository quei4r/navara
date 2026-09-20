/**
 * TEMP DEBUG PAGE (not committed): polyline layer A/B for the WebGPU TSL
 * port (PolylineMesh.initWebGPUMaterial). `?classic` uses the stock
 * WebGLRenderer path as reference.
 *
 * Loads the LACS MVT buildings (context) + a local debug GeoJSON of
 * polylines (zigzag joints, loop, long lines, a horizon-crossing line) with
 * a per-feature evaluator driving batch color/width.
 */
import ThreeView, { Color } from "@navaramap/three";
import {
  DefaultPlugin,
  type DefaultDescriptions,
} from "@navaramap/three-default-plugin";
import type { WebGLRenderer } from "three";
import { WebGPURenderer } from "three/webgpu";

import lines from "./lines.json";

const PALETTE = ["#ff3355", "#33ccff", "#ffcc33", "#66ff88", "#cc66ff"];

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

  const view = new ThreeView<DefaultDescriptions>({
    canvas,
    renderer: renderer as unknown as WebGLRenderer,
    animation: true,
  });
  view.addPlugin(new DefaultPlugin());
  await view.init();

  // Non-clamped polylines render through a lit shader; without a light they
  // draw black (same note as debug/geometry-types).
  view.addLight({ ambient: { intensity: 1 } });
  // ?sun — exercise the CSM (CascadedDirectionalLights) path; on WebGPU the
  // orchestrator mirrors it as a single directional proxy.
  if (query.has("sun")) {
    view.addLight({ sun: {} });
  }

  // Noon sun.
  const noon = new Date();
  noon.setUTCHours(4, 0, 0, 0);
  view.atmosphere.date = noon;

  // Esri imagery base (the classic path renders black without a raster
  // layer — same存量 quirk as the other TEMP pages).
  const ESRI_IMAGERY =
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
  const imagery = view.addSource({
    type: "raster-tile",
    url: ESRI_IMAGERY,
    maxZoom: 19,
  });
  view.addLayer({ type: "raster", source: imagery });

  // LACS MVT buildings for context (polygon WebGPU path, already ported).
  if (!query.has("nobld")) {
    const mvt = view.addSource({
      type: "vector-tile",
      url: "http://localhost:8787/tiles/{z}/{x}/{y}.mvt",
      maxZoom: 15,
    });
    view.addLayer({
      type: "vector",
      source: mvt,
      sourceLayers: ["building"],
      polygon: {
        color: new Color().setStyle("#9fb3c8"),
        clampToGround: false,
        height: 0,
        extrudedHeight: 60,
      },
    });
  }

  // Debug polylines (inline data — the WASM worker cannot resolve the
  // relative public-asset URL).
  const linesSrc = view.addSource({
    type: "geojson",
    data: lines,
  } as Parameters<typeof view.addSource>[0]);
  const layer = view.addLayer({
    type: "vector",
    source: linesSrc,
    polyline: {
      show: true,
      color: new Color().setStyle("#ff3355"),
      width: 6,
      height: 30,
      // Force the 3D shadow-volume polyline path (the WebGPU TSL port under
      // test): clampToGround + tile coordinates would bake to draped tiles.
      clampToGround: false,
    },
  });
  // Per-feature batch color + line width (exercises COLOR_SHOW / LINE_WIDTH
  // batch texture rows through the evaluator).
  layer.on("featureUpdated", ({ evaluator }) => {
    evaluator.evaluate(({ properties }) => {
      const idx = (properties?.["idx"] as number) ?? 0;
      return {
        color: new Color().setStyle(PALETTE[idx % PALETTE.length]),
        width: 4 + (idx % 3) * 5,
      };
    });
  });

  view.setCamera({
    lng: 118.0894,
    lat: 24.4455,
    height: 1500,
    heading: 0,
    pitch: -50,
    roll: 0,
  });

  (window as unknown as Record<string, unknown>).__lineDebug = {
    ready: true,
    view,
    renderer,
    // Inspect which vertex path the polyline meshes took (RTE vs plain) and
    // which material class they render with.
    polylineInfo: () => {
      const scenes =
        (
          view as unknown as {
            viewContext?: { scenes?: Record<string, unknown> };
          }
        ).viewContext?.scenes ?? {};
      const out: Record<string, unknown>[] = [];
      for (const [name, scene] of Object.entries(scenes)) {
        if (!scene || typeof scene !== "object" || !("traverse" in scene)) {
          continue;
        }
        (scene as { traverse: (cb: (o: never) => void) => void }).traverse(
          (o: {
            constructor?: { name?: string };
            geometryType?: string;
            geometry?: { attributes?: Record<string, unknown> };
            material?: { type?: string; userData?: Record<string, unknown> };
          }) => {
            const isPolyline =
              o.geometryType === "polyline" ||
              o.geometry?.attributes?.[
                "right_normal_and_texture_coordinate_normalization_y"
              ] != null;
            if (!isPolyline) return;
            out.push({
              scene: name,
              ctor: o.constructor?.name,
              geometryType: o.geometryType,
              matType: o.material?.type,
              webgpu: !!o.material?.userData?.nvrWebgpu,
              attrs: o.geometry?.attributes
                ? Object.keys(o.geometry.attributes)
                : [],
            });
          },
        );
      }
      return out;
    },
  };
  view.forceUpdate();
};

bootstrap().catch((e) => {
  console.error(e);
  const div = document.createElement("div");
  div.style.cssText =
    "position:fixed;inset:0;color:#f66;font:14px monospace;white-space:pre-wrap;padding:2em;";
  div.textContent = String((e as Error)?.stack ?? e);
  document.body.appendChild(div);
});
