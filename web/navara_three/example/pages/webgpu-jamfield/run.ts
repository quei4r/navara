// GPU 干扰场（DPM 波前 + 方块壳 + 地形贴合热图）Navara 示例页主逻辑。
// 移植自 LACS r185 页面的 jamfield 部分（不含火焰）：每源三层 JamField
// （低/中/高档频段分层），compute/TSL 逻辑逐字照抄，只改坐标参数化与对接层。
import ThreeView, {
  MeshDesc,
  PickableMeshWrapper,
  TERRARIUM_ELEVATION_DECODER,
  vector3ToGeodetic,
  type MeshConfig,
  type MeshHandle,
  type MeshUpdate,
  type PassKey,
  type ViewContext,
} from "@navaramap/three";
import {
  Group,
  Raycaster,
  SphereGeometry,
  Vector2,
  Vector3,
  Mesh,
  type Material,
} from "three";
import * as TSL from "three/tsl";
import { MeshBasicNodeMaterial, type WebGPURenderer } from "three/webgpu";

import { TERRAIN_DATASETS } from "../../helpers/constants";
import { atZoneTime } from "../../helpers/control";

import { makeLocalCoords } from "./coords";
import { createHeatMesh, type HeatMesh } from "./heatmesh";
import { BANDS, BAND_TIERS } from "./jambands";
import { createJamField, type JamField } from "./jamfield";
import { buildObstacleField, type ObstacleField } from "./obstacles";

const ESRI_IMAGERY =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

const HOME = { lng: 118.083, lat: 24.461 }; // 厦门
const MAX_JAM = 4;
const JAM_COLORS = [0x66eeff, 0xffaa44, 0xff66aa, 0x88ff88];
const BOX_Y = 160; // 盒 y 中心统一 160m（r185 拍板值）

// 干扰场三档分层（2026-09-11 拍板分层，2026-09-14 随 score=0 半径重标定收缩，
// 参数照抄 r185 main.ts 的 JAM_TIERS，勿改）：
// 盒尺寸按档内最远频段介入半径+10% 定 → 场在盒内自然归零，方边根除。
// 低档 z13 Terrarium 纯地形（低频可穿墙，白膜遮挡省掉）；中档 z13+MVT 建筑；
// 高档 z15 全遮挡细盒。
type JamTierCfg = {
  name: string;
  bands: number[];
  boxSize: [number, number, number];
  grid: [number, number, number];
  tileZ: number;
  buildings: boolean;
  maxDistM: number;
  expandFrames: number;
  fadeXZm: number;
};
const JAM_TIERS: JamTierCfg[] = [
  {
    name: "low",
    bands: BAND_TIERS[0],
    boxSize: [5500, 1000, 5500],
    grid: [220, 40, 220],
    tileZ: 13,
    buildings: false,
    maxDistM: 2750,
    expandFrames: 160,
    fadeXZm: 100,
  },
  {
    name: "mid",
    bands: BAND_TIERS[1],
    boxSize: [4400, 600, 4400],
    grid: [352, 48, 352],
    tileZ: 13,
    buildings: true,
    maxDistM: 2200,
    expandFrames: 240,
    fadeXZm: 100,
  },
  {
    name: "high",
    bands: BAND_TIERS[2],
    boxSize: [2640, 360, 2640],
    grid: [424, 64, 424],
    tileZ: 15,
    buildings: true,
    maxDistM: 1320,
    expandFrames: 320,
    fadeXZm: 200,
  },
];

type BuiltTier = {
  jam: JamField;
  obstacle: ObstacleField;
  cfg: JamTierCfg;
  boxMin: Vector3;
  boxSize: Vector3;
};

// ---------------------------------------------------------------------------
// MeshDesc：一源一组（rotY(π) 桥接 Group + 3 层壳 + 热图网格）
// Navara geodetic 局部帧是 West-Up-North（x西/y上/z北），r185 坐标是
// x东/y上/z南 —— 映射 = 绕 Y 转 π：desc 根下放中间 Group rotation.y=π，
// 子节点全部保持 r185 原生 ENU 坐标（米）。不用 scale(-1,1,-1)（翻转手性
// 影响面剔除）。
// marker 是独立的 JamMarkerMeshDesc（opaque）：PickHelper 会把 pickable raw
// 摘到独立 pickScene 渲染，raw 必须是 desc 根本体（matrixWorld 手动设置、
// matrixWorldAutoUpdate=false 重挂不丢变换）——包 Group 子节点会在 pick
// pass 里丢掉 ECEF 帧+rotY 桥而拣不中（已踩坑）。
// ---------------------------------------------------------------------------

type JamSourceDescription = {
  jamSource?: {
    tiers: BuiltTier[];
    color: number;
  };
};
export type JamSourceConfig = MeshConfig & JamSourceDescription;
export type JamSourceUpdate = MeshUpdate & JamSourceDescription;

export class JamSourceMeshDesc extends MeshDesc<
  JamSourceConfig,
  JamSourceUpdate,
  Group
> {
  readonly tiers: BuiltTier[];
  heatMesh?: HeatMesh;

  constructor(view: ThreeView, ctx: ViewContext, config: JamSourceConfig) {
    super(view, ctx, config);
    this.tiers = config.jamSource?.tiers ?? [];
  }

  protected override getPassKey(): PassKey {
    return "transparent";
  }

  createMesh(): Group {
    const root = new Group();
    root.name = "jam-source";
    const enu = new Group();
    enu.rotation.y = Math.PI; // WUN → ENU 桥（见文件头注释）
    root.add(enu);

    for (const t of this.tiers) enu.add(t.jam.mesh);

    this.heatMesh = createHeatMesh(
      this.tiers.map((t) => ({
        jam: t.jam,
        obstacle: t.obstacle,
        boxMin: t.boxMin,
        boxSize: t.boxSize,
      })),
    );
    enu.add(this.heatMesh.mesh);
    return root;
  }

  override onDestroy(): void {
    for (const t of this.tiers) {
      t.jam.dispose();
      t.obstacle.texture.dispose();
    }
    this.heatMesh?.dispose();
    super.onDestroy();
  }
}

// ---------------------------------------------------------------------------
// Marker desc（opaque，desc 根本体即球体——GPU picking 友好，见上注释）。
// geodetic.height 由调用方给贴地高度（细层 obstacle surfaceY + 6，椭球基准）。
// ---------------------------------------------------------------------------

type JamMarkerDescription = {
  jamMarker?: { color?: number };
};
export type JamMarkerConfig = MeshConfig & JamMarkerDescription;
export type JamMarkerUpdate = MeshUpdate & JamMarkerDescription;

export class JamMarkerMeshDesc extends MeshDesc<
  JamMarkerConfig,
  JamMarkerUpdate,
  Mesh
> {
  private cfg: JamMarkerConfig;
  private pickable?: PickableMeshWrapper;

  constructor(view: ThreeView, ctx: ViewContext, config: JamMarkerConfig) {
    super(view, ctx, config);
    this.cfg = config;
  }

  get batchId(): number | undefined {
    return this.pickable?.batchId;
  }

  createMesh(): Mesh {
    const mesh = new Mesh(
      new SphereGeometry(14, 20, 14),
      new MeshBasicNodeMaterial({
        color: this.cfg.jamMarker?.color ?? 0xffffff,
      }),
    );
    mesh.name = "jam-marker";
    this.pickable = new PickableMeshWrapper(mesh, this.ctx);
    this.ctx.registerPickableMesh(this.id, this.pickable);
    return mesh;
  }

  override onDestroy(): void {
    if (this.pickable) {
      this.ctx.unregisterPickableMesh(this.id);
      this.pickable = undefined;
    }
    if (this._instance) {
      this._instance.geometry.dispose();
      (this._instance.material as Material).dispose();
    }
    super.onDestroy();
  }
}

export type CustomDescriptions = {
  mesh: JamSourceConfig | JamMarkerConfig;
};

// ---------------------------------------------------------------------------
// Page logic
// ---------------------------------------------------------------------------

type JamNode = {
  id: number;
  lng: number;
  lat: number;
  handle: MeshHandle<JamSourceMeshDesc>;
  markerHandle: MeshHandle<JamMarkerMeshDesc>;
  tiers: BuiltTier[];
  baseColor: number;
  bandMask: number; // 7 位频段开关（默认全开）
  opacity: number; // 面板总透明度 0..1
};

export const run = async (
  view: ThreeView<CustomDescriptions>,
  renderer: WebGPURenderer | undefined,
) => {
  await view.init();
  view.registerMesh("jamSource", JamSourceMeshDesc);
  view.registerMesh("jamMarker", JamMarkerMeshDesc);

  // 上午光照，卫星底图清晰可读
  view.atmosphere.date = atZoneTime(
    view.atmosphere.date,
    10,
    0,
    "Asia/Shanghai",
  );

  // Esri 卫星底图 + mapterhorn DEM 地形层（Terrarium 编码，与障碍场采样基准一致）
  const imagery = view.addSource({
    type: "raster-tile",
    url: ESRI_IMAGERY,
    maxZoom: 19,
  });
  view.addLayer({ type: "raster", source: imagery });
  const dem = view.addSource({
    type: "raster-dem",
    url: TERRAIN_DATASETS.mapterhorn.url,
    maxZoom: 17,
    minZoom: 5,
    elevationDecoder: TERRARIUM_ELEVATION_DECODER(),
    tileSize: 512,
  });
  view.addLayer({ type: "terrain", source: dem, terrain: {} });

  // 机位放在源点南侧看北（Navara setCamera 是把相机放在 (lng,lat,height)
  // 再按 heading/pitch 瞄准，不是"看向该点"——直接给源点坐标会把源压在
  // 画面底边外，已实测）：源在 ~2km 外、俯角 ~47° 时源心居中
  view.setCamera({
    lng: HOME.lng,
    lat: HOME.lat - 0.018, // ~2km 南
    height: 2000,
    heading: 0,
    pitch: -47,
    roll: 0,
  });

  // ---------- HUD / UI ----------
  const hud = document.createElement("div");
  hud.style.cssText =
    "position:fixed;left:12px;bottom:12px;padding:8px 12px;background:rgba(0,0,0,.75);color:#7f7;font:13px monospace;white-space:pre;z-index:10;pointer-events:none;";
  document.body.appendChild(hud);
  const hudLines: string[] = [];
  const setHud = (key: string, text: string) => {
    const idx = hudLines.findIndex((l) => l.startsWith(key));
    const line = `${key}${text}`;
    if (idx >= 0) hudLines[idx] = line;
    else hudLines.push(line);
    hud.textContent = hudLines.join("\n");
  };

  const bar = document.createElement("div");
  bar.style.cssText =
    "position:fixed;left:12px;top:12px;display:flex;gap:8px;z-index:10;";
  document.body.appendChild(bar);
  const mkBtn = (label: string) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText =
      "padding:6px 12px;background:rgba(0,0,0,.7);color:#fff;border:1px solid #666;border-radius:4px;font:14px sans-serif;cursor:pointer;";
    bar.appendChild(b);
    return b;
  };
  const addJamBtn = mkBtn("➕ 添加干扰机");
  const delJamBtn = mkBtn("🗑 删除干扰机");
  delJamBtn.disabled = true;
  const MODE_LABELS = ["🧊 方块壳", "🌡 覆盖热图", "🧊🌡 叠加"];
  const renderBtn = mkBtn(MODE_LABELS[0]);

  // 选中面板：7 频段开关（勾掉只重合成不重洪泛）+ 总透明度（壳 uniform + 热图 uniform）
  const panel = document.createElement("div");
  panel.style.cssText =
    "position:fixed;right:12px;top:12px;padding:10px 14px;background:rgba(0,0,0,.8);color:#eee;font:13px sans-serif;z-index:10;display:none;border-radius:4px;";
  document.body.appendChild(panel);
  const panelTitle = document.createElement("div");
  panelTitle.style.cssText = "font-weight:bold;margin-bottom:6px;";
  panel.appendChild(panelTitle);
  const bandChecks: HTMLInputElement[] = [];
  BANDS.forEach((b, bi) => {
    const label = document.createElement("label");
    label.style.cssText =
      "display:flex;align-items:center;gap:6px;margin:2px 0;";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => {
      if (!selectedJam) return;
      selectedJam.bandMask = cb.checked
        ? selectedJam.bandMask | (1 << bi)
        : selectedJam.bandMask & ~(1 << bi);
      for (const t of selectedJam.tiers) t.jam.setBandEnable(0, bi, cb.checked);
    });
    const chip = document.createElement("span");
    chip.style.cssText = `display:inline-block;width:12px;height:12px;border-radius:2px;background:rgb(${BANDS[bi].color.map((v) => Math.round(v * 255)).join(",")});`;
    label.append(cb, chip, document.createTextNode(b.name));
    panel.appendChild(label);
    bandChecks.push(cb);
  });
  const opRow = document.createElement("div");
  opRow.style.cssText =
    "margin-top:8px;display:flex;align-items:center;gap:6px;";
  const opSlider = document.createElement("input");
  opSlider.type = "range";
  opSlider.min = "0";
  opSlider.max = "100";
  opSlider.value = "100";
  const opVal = document.createElement("span");
  opVal.textContent = "100%";
  opSlider.addEventListener("input", () => {
    if (!selectedJam) return;
    const v = parseInt(opSlider.value, 10) / 100;
    selectedJam.opacity = v;
    opVal.textContent = `${opSlider.value}%`;
    for (const t of selectedJam.tiers) t.jam.setOpacity(v);
    selectedJam.handle.ref.heatMesh?.setOpacity(v);
  });
  opRow.append(document.createTextNode("透明度"), opSlider, opVal);
  panel.appendChild(opRow);

  // ---------- 干扰机管理 ----------
  const jamNodes: JamNode[] = [];
  const byBatchId = new Map<number, JamNode>();
  let jamNextId = 0;
  let addingJam = false;
  let selectedJam: JamNode | null = null;

  const syncJamPanel = () => {
    panel.style.display = selectedJam ? "block" : "none";
    if (!selectedJam) return;
    panelTitle.textContent = `干扰机 #${selectedJam.id}`;
    bandChecks.forEach((cb, bi) => {
      cb.checked = ((selectedJam?.bandMask ?? 0) & (1 << bi)) !== 0;
    });
    opSlider.value = String(Math.round(selectedJam.opacity * 100));
    opVal.textContent = `${Math.round(selectedJam.opacity * 100)}%`;
  };

  const selectJam = (node: JamNode | null) => {
    if (selectedJam) {
      const m = selectedJam.markerHandle.ref.raw;
      if (m)
        (m.material as MeshBasicNodeMaterial).color.setHex(
          selectedJam.baseColor,
        );
      selectedJam.markerHandle.update({ geodetic: { scale: 1 } });
    }
    selectedJam = node;
    if (node) {
      const m = node.markerHandle.ref.raw;
      if (m) (m.material as MeshBasicNodeMaterial).color.setHex(0xffffff);
      node.markerHandle.update({ geodetic: { scale: 1.5 } });
    }
    delJamBtn.disabled = !node;
    syncJamPanel();
  };

  let renderMode = 0;
  const applyRenderMode = (m: number) => {
    renderMode = ((m % 3) + 3) % 3;
    renderBtn.textContent = MODE_LABELS[renderMode];
    const shellVisible = renderMode !== 1;
    const heatVisible = renderMode !== 0;
    for (const n of jamNodes) {
      for (const t of n.tiers) t.jam.mesh.visible = shellVisible;
      const hm = n.handle.ref.heatMesh;
      if (hm) hm.mesh.visible = heatVisible;
    }
    setHud("mode: ", MODE_LABELS[renderMode]);
  };
  renderBtn.addEventListener("click", () => applyRenderMode(renderMode + 1));
  // ?mode=1|2：起始即热图/叠加（验证管线编译用）
  const modeParam = parseInt(
    new URLSearchParams(window.location.search).get("mode") ?? "",
    10,
  );

  const addJamNode = async (
    lng: number,
    lat: number,
  ): Promise<number | null> => {
    if (addingJam) return null;
    // Jam compute passes need the WebGPURenderer; `?classic` is render-only.
    if (!renderer) {
      setHud("jam: ", "classic 模式不支持干扰机");
      return null;
    }
    if (jamNodes.length >= MAX_JAM) {
      setHud("jam: ", `已达上限 ${MAX_JAM} 源`);
      return null;
    }
    addingJam = true;
    // 一源一 ENU 原点（盒心即源心局部 (0,?,0)）；内部 y 全部用 Terrarium
    // 绝对高程米 —— 与 Navara 地形（椭球+Terrarium）基准一致
    const coords = makeLocalCoords(lng, lat);
    const tiers: BuiltTier[] = [];
    try {
      for (let ti = 0; ti < JAM_TIERS.length; ti++) {
        const cfg = JAM_TIERS[ti];
        setHud(
          "jam: ",
          `建障碍场 ${cfg.name} (${ti + 1}/${JAM_TIERS.length})…`,
        );
        const obstacle = await buildObstacleField(
          coords,
          new Vector3(0, BOX_Y, 0),
          cfg.boxSize,
          cfg.grid,
          cfg.tileZ,
          { buildings: cfg.buildings },
        );
        const boxMin = new Vector3(
          -cfg.boxSize[0] / 2,
          BOX_Y - cfg.boxSize[1] / 2,
          -cfg.boxSize[2] / 2,
        );
        const boxSize = new Vector3(...cfg.boxSize);
        const jam = createJamField(
          renderer,
          obstacle,
          cfg.grid,
          boxMin,
          boxSize,
          [{ x: 0, z: 0 }],
          {
            bands: cfg.bands,
            maxDistM: cfg.maxDistM,
            expandFrames: cfg.expandFrames,
            fadeXZm: cfg.fadeXZm,
          },
        );
        tiers.push({ jam, obstacle, cfg, boxMin, boxSize });
      }
      const baseColor = JAM_COLORS[jamNextId % JAM_COLORS.length];
      const handle = view.addMesh<JamSourceMeshDesc>({
        geodetic: { lng, lat, height: 0, heightReference: "ellipsoid" },
        jamSource: { tiers, color: baseColor },
      });
      // marker 贴地高度用细层（最精确含建筑），椭球基准
      const groundY = tiers[tiers.length - 1].obstacle.surfaceY(0, 0);
      const markerHandle = view.addMesh<JamMarkerMeshDesc>({
        geodetic: {
          lng,
          lat,
          height: groundY + 6,
          heightReference: "ellipsoid",
        },
        jamMarker: { color: baseColor },
      });
      const node: JamNode = {
        id: jamNextId++,
        lng,
        lat,
        handle,
        markerHandle,
        tiers,
        baseColor,
        bandMask: 0x7f,
        opacity: 1,
      };
      jamNodes.push(node);
      if (markerHandle.ref.batchId !== undefined) {
        byBatchId.set(markerHandle.ref.batchId, node);
      }
      applyRenderMode(renderMode); // 新源继承当前渲染模式
      console.log(
        "[jam] added @",
        lng.toFixed(4),
        lat.toFixed(4),
        `(${jamNodes.length}/${MAX_JAM})`,
      );
      return node.id;
    } catch (err) {
      console.error("[jam] add failed", err);
      for (const t of tiers) {
        t.jam.dispose();
        t.obstacle.texture.dispose();
      }
      setHud("jam: ", "添加失败（见 console）");
      return null;
    } finally {
      addingJam = false;
    }
  };

  const removeJamNode = (node: JamNode) => {
    if (selectedJam === node) selectJam(null);
    if (node.markerHandle.ref.batchId !== undefined) {
      byBatchId.delete(node.markerHandle.ref.batchId);
    }
    node.markerHandle.delete();
    node.handle.delete(); // desc.onDestroy 里 dispose 各 JamField + 障碍纹理 + 热图
    jamNodes.splice(jamNodes.indexOf(node), 1);
    console.log("[jam] removed #", node.id, `(${jamNodes.length}/${MAX_JAM})`);
  };

  // 「➕」armed 后点地面放置（短促点击才触发，区分拖拽）
  let addArmed = false;
  const disarm = () => {
    addArmed = false;
    addJamBtn.textContent = "➕ 添加干扰机";
    addJamBtn.style.background = "rgba(0,0,0,.7)";
  };
  addJamBtn.addEventListener("click", () => {
    addArmed = !addArmed;
    addJamBtn.textContent = addArmed ? "📍 点击地面放置" : "➕ 添加干扰机";
    addJamBtn.style.background = addArmed ? "#2a6" : "rgba(0,0,0,.7)";
  });
  delJamBtn.addEventListener("click", () => {
    if (selectedJam) removeJamNode(selectedJam);
  });

  // 点 marker = 选中/再点取消（GPU picking，batchId → node）
  view.on("featureClick", (info) => {
    const node = info ? byBatchId.get(info.batchId) : undefined;
    if (node) selectJam(selectedJam === node ? null : node);
  });

  // 放置取点：view 的 click 事件自带 map=ECEF 命中点（convertScreenToWorld
  // 走 WASM 的 WGS84 椭球求交，纯数学与渲染后端无关——WebGPU 下
  // pickDepthPosition 不可用；自算 three Raycaster 在 Navara 相机封装下
  // 不可靠，实测把源放到了 5800km 外）。天空/地平线以上点击 map 为空，
  // click 事件根本不触发，天然忽略。
  const canvas = renderer?.domElement ?? view.canvas;
  view.on("click", (e) => {
    if (!addArmed) return;
    const geo = vector3ToGeodetic(new Vector3(e.map.x, e.map.y, e.map.z));
    disarm();
    void addJamNode(geo.lng, geo.lat);
  });

  // ---------- 帧驱动（busy 防重入；compute 只按预算推进、收敛即全停） ----------
  let lastT: number | null = null;
  let busy = false;
  let frames = 0;
  view.on("preUpdate", (t: number) => {
    if (busy) return;
    const dt = lastT === null ? 1 / 60 : Math.min((t - lastT) / 1000, 0.1);
    lastT = t;
    busy = true;
    void (async () => {
      for (const n of jamNodes) {
        for (const tier of n.tiers) await tier.jam.update(dt, view.camera.raw);
      }
      // 收敛前 display 纹理会 ping-pong 换对象，热图材质每帧同步
      for (const n of jamNodes) n.handle.ref.heatMesh?.refreshTextures();
      frames++;
      if (frames === 1) console.log("[jam] FIRST_FRAME");
      if (frames % 30 === 0) {
        const conv = jamNodes.filter((n) =>
          n.tiers.every((t) => t.jam.converged()),
        ).length;
        setHud(
          "jam: ",
          `${jamNodes.length}/${MAX_JAM} 源, ${conv} 收敛${selectedJam ? `, 选中 #${selectedJam.id}` : ""}`,
        );
      }
    })()
      .catch((e) => {
        console.error("[jam] frame error", e);
        setHud("ERR: ", String(e?.message ?? e).slice(0, 200));
      })
      .finally(() => {
        busy = false;
      });
  });

  // 初始源：默认厦门（?jam=lng,lat;... 自定义，?jam=0 不起初始源）
  const q = new URLSearchParams(window.location.search);
  const jamQ = q.get("jam");
  if (jamQ !== "0") {
    const srcLLs = (jamQ && jamQ !== "1" ? jamQ.split(";") : [])
      .map((s) => s.split(",").map(parseFloat))
      .filter((a) => a.length === 2 && a.every((v) => isFinite(v)));
    const srcs = srcLLs.length ? srcLLs : [[HOME.lng, HOME.lat]];
    for (const [lng, lat] of srcs.slice(0, MAX_JAM)) await addJamNode(lng, lat);
  }
  if (modeParam >= 1 && modeParam <= 2) applyRenderMode(modeParam);

  // ---------- 调试接口 ----------
  const screenPosOf = (id: number) => {
    const node = jamNodes.find((n) => n.id === id);
    const marker = node?.markerHandle.ref.raw;
    if (!node || !marker) return null;
    const wp = marker.getWorldPosition(new Vector3());
    const ndc = wp.project(view.camera.raw);
    const rect = canvas.getBoundingClientRect();
    return {
      x: rect.left + (ndc.x * 0.5 + 0.5) * rect.width,
      y: rect.top + (1 - (ndc.y * 0.5 + 0.5)) * rect.height,
    };
  };
  (window as unknown as Record<string, unknown>).__jamDebug = {
    ready: true,
    view,
    renderer,
    addJam: (lng: number, lat: number) => addJamNode(lng, lat),
    removeJam: (id: number) => {
      const n = jamNodes.find((v) => v.id === id);
      if (n) removeJamNode(n);
    },
    getNodes: () =>
      jamNodes.map((n) => ({
        id: n.id,
        lng: n.lng,
        lat: n.lat,
        converged: n.tiers.every((t) => t.jam.converged()),
        selected: selectedJam === n,
        batchId: n.markerHandle.ref.batchId,
      })),
    setMode: (m: number) => applyRenderMode(m),
    screenPosOf,
    tsl: TSL,
    raycastAt: (x: number, y: number) => {
      const rect = canvas.getBoundingClientRect();
      const ndc = new Vector2(
        ((x - rect.left) / rect.width) * 2 - 1,
        -(((y - rect.top) / rect.height) * 2 - 1),
      );
      const rc = new Raycaster();
      rc.setFromCamera(ndc, view.camera.raw);
      const out: {
        scene: string;
        matType?: string;
        hasMap?: boolean;
        hasColorAttr?: boolean;
        colorSample?: number[];
        geoAttrs?: string[];
        matUuid?: string;
        dist?: number;
      }[] = [];
      const scenes =
        (
          view as unknown as {
            viewContext?: { scenes?: Record<string, unknown> };
          }
        ).viewContext?.scenes ?? {};
      for (const [name, scene] of Object.entries(scenes)) {
        if (!scene || typeof scene !== "object" || !("traverse" in scene)) {
          continue;
        }
        const hits = rc.intersectObject(scene as never, true) as unknown as {
          object: Mesh;
          distance: number;
        }[];
        for (const h of hits.slice(0, 3)) {
          const o = h.object;
          const m = o.material as unknown as {
            type?: string;
            map?: unknown;
            uuid?: string;
          };
          const c = o.geometry?.getAttribute?.("color");
          out.push({
            scene: name,
            matType: m?.type,
            hasMap: !!m?.map,
            hasColorAttr: !!c,
            colorSample: c ? [c.getX(0), c.getY(0), c.getZ(0)] : undefined,
            geoAttrs: o.geometry ? Object.keys(o.geometry.attributes) : [],
            matUuid: m?.uuid,
            dist: Math.round(h.distance),
          });
        }
      }
      return out;
    },
  };
  view.forceUpdate();
};
