/**
 * TEMP DEBUG PAGE (not committed): model enhancer FX + pnts A/B for the
 * WebGPU TSL port (ModelMesh._convertMeshToNodeMaterial /
 * _convertPointsToNodeMaterial). No flag = WebGPU, `?classic` = stock
 * WebGLRenderer reference.
 *
 * `?fx=none|specular|water|emissive` picks the b3dm model effect under test:
 *  - specular: sun glint via the specular enhancer props
 *  - water: animated water normal/specular via the water enhancer props
 *  - emissive: selective-effect emissive (selectiveBloom registry slot +
 *    effectIds + emissiveIntensity)
 * `?pnts` swaps the b3dm layer for the synthetic pnts tileset in
 * public/pnts-debug (point cloud path); `?height=300` offsets the points
 * along the geodetic normal.
 */
import ThreeView, { Color } from "@navaramap/three";
import {
  DefaultPlugin,
  type DefaultDescriptions,
} from "@navaramap/three-default-plugin";
import type { WebGLRenderer } from "three";
import { WebGPURenderer } from "three/webgpu";

const B3DM_TILESET = "http://localhost:8787/city3d/xiamen/tileset.json";
const PNTS_TILESET = "/examples/pnts-debug/tileset.json";
// Known-good PLATEAU point cloud (official point-cloud example dataset).
const KAKEGAWA_TILESET =
  "https://assets.cms.plateau.reearth.io/assets/6b/68c785-f43d-4451-ba7f-d4d130ef6ba5/uc_pv1_22213_kakegawa/pointcloud/22213_kakegawa_castle/tileset.json";

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
  const fx = query.get("fx") ?? "none";
  const usePnts = query.has("pnts");
  const height = Number(query.get("height") ?? 0);
  // `?bare` drops DefaultPlugin + imagery (bisect against webgpu-b3dm).
  const bare = query.has("bare");
  // `?unlit` mirrors webgpu-b3dm's layer config exactly (model: {lit:false}).
  const unlit = query.has("unlit");

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
    // Shared water normal texture (CommonUniforms.waterTexture).
    waterTexture: { enabled: fx === "water" },
  });
  if (!bare) view.addPlugin(new DefaultPlugin());
  await view.init();

  // App lights: classic path has no fallback rig — lit materials render
  // black without these. WebGPU mirrors them via light proxies. castShadow
  // off so the fx A/B isn't dominated by CSM vs WebGPUEnvironment shadow
  // differences.
  view.addLight({ ambient: { intensity: 0.6 } } as never);
  view.addLight({ sun: { intensity: 1, castShadow: false } } as never);

  // Noon sun over Xiamen.
  const noon = new Date();
  noon.setUTCHours(4, 0, 0, 0);
  view.atmosphere.date = noon;

  // Esri imagery base so the classic path isn't black.
  if (!bare) {
    const ESRI_IMAGERY =
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
    const imagery = view.addSource({
      type: "raster-tile",
      url: ESRI_IMAGERY,
      maxZoom: 19,
    });
    view.addLayer({ type: "raster", source: imagery });
  }

  if (usePnts) {
    // `?kakegawa` swaps in the known-good PLATEAU point cloud instead of the
    // synthetic local tileset.
    const src = view.addSource({
      type: "3d-tiles",
      url: query.has("kakegawa") ? KAKEGAWA_TILESET : PNTS_TILESET,
    });
    view.addLayer({
      type: "3d-tiles",
      source: src,
      model: {
        color: new Color().setHex(0xff3322),
        pointSize: 6,
        maxSse: 8,
        height,
      },
    });
  } else {
    const model: Record<string, unknown> = unlit
      ? { lit: false }
      : {
          color: new Color().setHex(0xffffff),
          metalness: 0,
          roughness: 1,
        };
    if (fx === "specular") {
      model.specular = true;
      model.shininess = 64;
      model.specularStrength = 3;
      model.ior = 1.5;
    } else if (fx === "water") {
      model.water = true;
      model.waterScaleNormal = 0.02;
      model.waterSpeed = 0.001;
      model.shininess = 64;
      model.specularStrength = 3;
    } else if (fx === "emissive") {
      const bloom = view.addEffect({
        selectiveBloom: { strength: 0.6, radius: 0.4, threshold: 0.4 },
      } as Parameters<typeof view.addEffect>[0]);
      model.effectIds = [bloom.id];
      model.emissiveColor = new Color().setHex(0xff2200);
      model.emissiveIntensity = 0.5;
    }
    const src = view.addSource({ type: "3d-tiles", url: B3DM_TILESET });
    view.addLayer({
      type: "3d-tiles",
      source: src,
      model: model as never,
    });
  }

  view.setCamera({
    lng: Number(query.get("lng") ?? 118.075),
    lat: Number(query.get("lat") ?? 24.457),
    height: Number(query.get("h") ?? (usePnts ? 800 : 1600)),
    heading: Number(query.get("heading") ?? 0),
    pitch: Number(query.get("pitch") ?? -45),
    roll: 0,
  });

  (window as unknown as Record<string, unknown>).__fxDebug = {
    ready: true,
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
