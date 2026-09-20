/**
 * TEMP DEBUG PAGE (not committed): clamp-to-ground (draped) vector A/B for
 * the WebGPU tile-drape bake path (TileTextureCompositor.renderVectorScenes +
 * VectorDrapeResolver node-slot binding). `?classic` uses the stock
 * WebGLRenderer path as reference.
 *
 * Setup: Esri imagery + mapterhorn DEM terrain (drape conforms to relief),
 * LACS MVT buildings with clampToGround: true (draped polygons, orange),
 * and the webgpu-polyline debug GeoJSON with clampToGround: true (draped
 * polylines, per-feature palette/width).
 */
import ThreeView, {
  Color,
  TERRARIUM_ELEVATION_DECODER,
} from "@navaramap/three";
import {
  DefaultPlugin,
  type DefaultDescriptions,
} from "@navaramap/three-default-plugin";
import type { WebGLRenderer } from "three";
import { WebGPURenderer } from "three/webgpu";

import { TERRAIN_DATASETS } from "../../helpers/constants";
import lines from "../webgpu-polyline/lines.json";

const PALETTE = ["#ff3355", "#33ccff", "#ffcc33", "#66ff88", "#cc66ff"];

const ESRI_IMAGERY =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

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

  // Noon sun.
  const noon = new Date();
  noon.setUTCHours(4, 0, 0, 0);
  view.atmosphere.date = noon;

  const imagery = view.addSource({
    type: "raster-tile",
    url: ESRI_IMAGERY,
    maxZoom: 19,
  });
  view.addLayer({ type: "raster", source: imagery });

  if (!query.has("noterrain")) {
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

  // Draped MVT buildings (clamp-to-ground polygons — the vector bake path).
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
        color: new Color().setStyle("#ff8c1a"),
        clampToGround: true,
      },
    });
  }

  // Draped debug polylines (same data as webgpu-polyline, but clamped).
  if (!query.has("noline")) {
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
        clampToGround: true,
      },
    });
    layer.on("featureUpdated", ({ evaluator }) => {
      evaluator.evaluate(({ properties }) => {
        const idx = (properties?.["idx"] as number) ?? 0;
        return {
          color: new Color().setStyle(PALETTE[idx % PALETTE.length]),
          width: 4 + (idx % 3) * 5,
        };
      });
    });
  }

  view.setCamera({
    lng: 118.0894,
    lat: 24.4455,
    height: 1500,
    heading: 0,
    pitch: -50,
    roll: 0,
  });

  (window as unknown as Record<string, unknown>).__drapeDebug = {
    ready: true,
    view,
    renderer,
    // Inspect tile materials: on WebGPU each tile node material carries
    // webgpuSlots (has/node uniforms); count active slots per region
    // (raster [0, from), vector [from, max)) and whether vector slots sample
    // a baked render-target texture.
    drapeInfo: () => {
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
            texturizedSceneIndexFrom?: number;
            maxTextures?: number;
            material?: {
              type?: string;
              userData?: Record<string, unknown>;
            };
          }) => {
            const slots = o.material?.userData?.webgpuSlots as
              | {
                  has: { value: number };
                  node: { value: { isRenderTargetTexture?: boolean } };
                }[]
              | undefined;
            const hasVd =
              "vectorDrape" in (o as unknown as Record<string, unknown>);
            if (!slots && !hasVd) return;
            const from = o.texturizedSceneIndexFrom ?? -1;
            const active: number[] = [];
            const vectorRT: number[] = [];
            (slots ?? []).forEach((s, i) => {
              if (s.has.value === 1) active.push(i);
              if (i >= from && s.has.value === 1) {
                vectorRT.push(s.node.value?.isRenderTargetTexture ? 1 : 0);
              }
            });
            const pos = {
              x: 0,
              y: 0,
              z: 0,
            };
            if ("matrixWorld" in o) {
              const e = (o.matrixWorld as { elements: number[] }).elements;
              pos.x = Math.round(e[12]);
              pos.y = Math.round(e[13]);
              pos.z = Math.round(e[14]);
            }
            const vd = (
              o as unknown as {
                vectorDrape?: {
                  vectorSlots?: {
                    layerId: string;
                    sources: {
                      tileHandle: bigint;
                      uvOffset: [number, number];
                      uvScale: [number, number];
                    }[];
                  }[];
                };
              }
            ).vectorDrape;
            const vslots = (vd?.vectorSlots ?? []).map((s) => ({
              layer: s.layerId,
              sources: s.sources.map((src) => ({
                h: String(src.tileHandle),
                off: src.uvOffset.map((v) => Math.round(v * 1e4) / 1e4),
                scale: src.uvScale.map((v) => Math.round(v * 1e4) / 1e4),
              })),
            }));
            out.push({
              scene: name,
              matType: o.material?.type,
              handle: "handle" in o ? String(o.handle) : undefined,
              pos,
              from,
              max: o.maxTextures,
              active,
              vectorRT,
              vslots,
            });
          },
        );
      }
      return out;
    },
    // Read back the nearest-to-camera tile's drape RT at slotIdx as a PNG
    // data URL — ground truth for what the bake drew (backend A/B).
    readRT: async (slotIdx: number, wantHandle?: string) => {
      const scenes =
        (
          view as unknown as {
            viewContext?: { scenes?: Record<string, unknown> };
          }
        ).viewContext?.scenes ?? {};
      const camPos = view.camera.positionECEF;
      let best: {
        d: number;
        handle: string;
        pos: { x: number; y: number; z: number };
        rt: unknown;
        slots: unknown;
      } | null = null;
      for (const scene of Object.values(scenes)) {
        if (!scene || typeof scene !== "object" || !("traverse" in scene)) {
          continue;
        }
        (scene as { traverse: (cb: (o: never) => void) => void }).traverse(
          (o: Record<string, unknown>) => {
            const vd = o.vectorDrape as
              | {
                  vectorSlots?: unknown;
                  renderTargets?: { width: number; height: number }[];
                }
              | undefined;
            if (!vd || !("handle" in o)) return;
            const rts = vd.renderTargets ?? [];
            if (!rts[slotIdx]) return;
            const e = (o.matrixWorld as { elements: number[] }).elements;
            const d = Math.hypot(
              e[12] - camPos.x,
              e[13] - camPos.y,
              e[14] - camPos.z,
            );
            const hit = wantHandle
              ? String(o.handle) === wantHandle
              : !best || d < best.d;
            if (hit) {
              best = {
                d,
                handle: String(o.handle),
                pos: {
                  x: Math.round(e[12]),
                  y: Math.round(e[13]),
                  z: Math.round(e[14]),
                },
                rt: rts[slotIdx],
                slots: (
                  vd.vectorSlots as {
                    layerId: string;
                    sources: {
                      tileHandle: bigint;
                      uvOffset: number[];
                      uvScale: number[];
                    }[];
                  }[]
                ).map((s) => ({
                  layer: s.layerId,
                  sources: s.sources.map((src) => ({
                    h: String(src.tileHandle),
                    off: src.uvOffset,
                    scale: src.uvScale,
                  })),
                })),
              };
            }
          },
        );
      }
      if (!best) return null;
      const b = best as {
        d: number;
        handle: string;
        pos: { x: number; y: number; z: number };
        rt: { width: number; height: number };
        slots: unknown;
      };
      const rt = b.rt;
      const w = rt.width;
      const h = rt.height;
      const r = (
        view as unknown as {
          _renderer: {
            isWebGPURenderer?: boolean;
            readRenderTargetPixelsAsync: (
              ...a: unknown[]
            ) => Promise<Uint8Array | undefined>;
          };
        }
      )._renderer;
      let buf: Uint8Array;
      if (r.isWebGPURenderer) {
        buf = (await r.readRenderTargetPixelsAsync(
          rt,
          0,
          0,
          w,
          h,
        )) as Uint8Array;
      } else {
        buf = new Uint8Array(w * h * 4);
        await r.readRenderTargetPixelsAsync(rt, 0, 0, w, h, buf);
      }
      const cv = document.createElement("canvas");
      cv.width = w;
      cv.height = h;
      const cx = cv.getContext("2d");
      if (!cx) return null;
      const im = cx.createImageData(w, h);
      im.data.set(buf);
      cx.putImageData(im, 0, 0);
      let nonzero = 0;
      for (let i = 3; i < buf.length; i += 4) if (buf[i] > 0) nonzero++;
      return {
        dataUrl: cv.toDataURL("image/png"),
        tile: b.handle,
        pos: b.pos,
        slots: b.slots,
        alphaFrac: nonzero / (w * h),
      };
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
