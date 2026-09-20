/**
 * TEMP DEBUG PAGE (not committed): sdfText + instancedSprite layer A/B for
 * the WebGPU TSL ports (BatchedSdfTextMesh._initWebGPUMaterial /
 * InstancedSpriteMesh._initWebGPUMaterial). `?classic` uses the stock
 * WebGLRenderer path as reference.
 *
 * One MultiPoint geojson drives three layers over the same anchors:
 *  - text with outline + background (SDF glyph / outline / background paths)
 *  - billboard sprites (atlas image path)
 *  - point sprites (circle path)
 * The LACS MVT buildings provide depth/occlusion context.
 */
import ThreeView, { Color, fetchFontFamilyFromCss } from "@navaramap/three";
import {
  DefaultPlugin,
  type DefaultDescriptions,
} from "@navaramap/three-default-plugin";
import type { WebGLRenderer } from "three";
import { WebGPURenderer } from "three/webgpu";

import { withBase } from "../../helpers/base";

const CENTER = { lng: 118.0894, lat: 24.4455 };

const ring = (d: number): number[][] => [
  [CENTER.lng - d, CENTER.lat - d],
  [CENTER.lng + d * 0.6, CENTER.lat - d * 0.7],
  [CENTER.lng - d * 0.5, CENTER.lat + d * 0.8],
  [CENTER.lng + d, CENTER.lat + d * 0.5],
  [CENTER.lng, CENTER.lat],
];

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
    webgpuDebug: {
      rawPost: query.has("rawpost"),
      noPost: query.has("nopost"),
    },
  });
  view.addPlugin(new DefaultPlugin());
  await view.init();

  view.addLight({ ambient: { intensity: 1 } });

  // Noon sun.
  const noon = new Date();
  noon.setUTCHours(4, 0, 0, 0);
  view.atmosphere.date = noon;

  view.addFontFamily(
    await fetchFontFamilyFromCss(
      "Arsenal",
      "https://fonts.googleapis.com/css2?family=Arsenal:wght@700",
    ),
  );

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

  // LACS MVT buildings for occlusion context (polygon WebGPU path, already
  // ported).
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

  const points = view.addSource({
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { name: "Alpha" },
          geometry: { type: "MultiPoint", coordinates: ring(0.004) },
        },
        {
          type: "Feature",
          properties: { name: "Beta" },
          geometry: { type: "MultiPoint", coordinates: ring(0.009) },
        },
      ],
    },
  } as Parameters<typeof view.addSource>[0]);

  // SDF text: outline + background, constant screen-pixel size (pxToWorld).
  const textLayer = view.addLayer({
    type: "vector",
    source: points,
    text: {
      text: "LACS-Label",
      font: "Arsenal",
      color: new Color().setStyle("#ffffff"),
      size: 42,
      sizeInMeters: false,
      clampToGround: false,
      height: 120,
      center: { x: 0.5, y: 0 },
      outlineColor: new Color().setStyle("#0033aa"),
      outlineWidth: 4,
      outlineOpacity: 1,
      backgroundColor: new Color().setStyle("#1a2a4a"),
      borderColor: new Color().setStyle("#ffcc33"),
      borderWidth: 0.08,
      declutter: false,
    },
  });
  // Per-feature text override (label data texture path).
  textLayer.on("featureUpdated", ({ evaluator }) => {
    evaluator.evaluate(({ properties }) => ({
      text: `站 ${(properties?.["name"] as string) ?? ""}`,
    }));
  });

  // Billboard sprites (atlas image path), world-sized.
  view.addLayer({
    type: "vector",
    source: points,
    billboard: {
      color: new Color().setStyle("#ffffff"),
      size: 90,
      sizeInMeters: true,
      clampToGround: false,
      height: 40,
      center: { x: 0, y: -0.5 },
      depthTest: true,
      alphaTest: 0.3,
      transparent: true,
      url: withBase("icons/restaurant.svg"),
      offsetDepth: true,
      declutter: false,
    },
  });

  // Point sprites (circle path), world-sized.
  view.addLayer({
    type: "vector",
    source: points,
    point: {
      size: 55,
      sizeInMeters: true,
      clampToGround: false,
      color: new Color().setStyle("#ff3355"),
      center: { x: 0, y: 0 },
      height: 20,
      offsetDepth: true,
      depthTest: true,
      transparent: true,
      declutter: false,
    },
  });

  view.setCamera({
    lng: CENTER.lng,
    lat: CENTER.lat,
    height: 1600,
    heading: 0,
    pitch: -50,
    roll: 0,
  });

  (window as unknown as Record<string, unknown>).__textSpriteDebug = {
    ready: true,
    view,
    renderer,
    // Inspect which material class the text/sprite meshes render with.
    meshInfo: () => {
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
            material?: { type?: string; userData?: Record<string, unknown> };
          }) => {
            const ctor = o.constructor?.name ?? "";
            if (!/SdfText|Sprite/i.test(ctor)) return;
            out.push({
              scene: name,
              ctor,
              geometryType: o.geometryType,
              matType: o.material?.type,
              webgpu: !!o.material?.userData?.nvrWebgpu,
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
