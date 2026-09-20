// 流式方向记忆波前干扰场（Dominant Path Model, GPU frontier 实现）
// 物理：dist = 波前等效距离（0.1m 单位 uint），直线段正常 FSPL 距离衰减；
//   每次转向乘 exp(λ·Δθ)（λ=ln4/(π/2)：90°×4≈12dB，180°×16≈24dB）。
//   建筑/地形体素不可穿透，波前只能沿街道/空域绕行 —— 小巷自动获得绕射式
//   覆盖，楼后保留大衰减阴影。公式移植自 Godot 仿真（见 jambands.ts）。
// 流式：frontier 波前每帧 K 轮持续松弛；源移动后 reset+重洪泛，
//   0.5-1s 内可见场从新位置扩散重收敛。
// 渲染（2026-09-11 拍板，弃 raymarch 雾）：Godot 同款**表面方块壳**——
//   emit compute 找"有场且 6 面盒内邻居至少一个空"的表面格，append 实例
//   （cellIndex+RGBA8），实例化立方体普通半透明混合直出。场强语义同 Godot
//   compute_unified_field/color_1m.glsl：每频段 alpha=0.1×clamp(score/30)，
//   饱和封顶+最高分选段（平局取高频）定色，alpha=选中频段场强。
//   无步进采样、无 DDA、遮挡由深度缓冲精确解决；收敛后壳几何静止零 compute。
import {
  vec3,
  vec4,
  float,
  int,
  uint,
  Fn,
  uniform,
  If,
  texture3D,
  textureStore,
  instanceIndex,
  smoothstep,
  mix,
  max,
  min,
  varying,
  storage,
  atomicStore,
  atomicLoad,
  atomicAdd,
  atomicMin,
  log2,
  exp,
  acos,
  dot,
  clamp,
  length,
  positionGeometry,
} from "three/tsl";
import * as THREE from "three/webgpu";

import {
  BANDS,
  GMAX_DBI,
  MARGIN_DBM,
  VERTICAL_N,
  packBandBuffer,
} from "./jambands";
import type { ObstacleField } from "./obstacles";

export type JamSourceEN = { x: number; z: number };
export type JamField = {
  mesh: THREE.Mesh;
  update: (dt: number, camera: THREE.Camera) => Promise<void>;
  moveSource: (i: number, en: JamSourceEN) => void;
  /** 体素盒整体平移到以 en 为圆心（y 锚定不变），随后自动重洪泛。
   *  调用前必须完成 obstacles.rebuild(en)，否则障碍采样与盒框架错位。 */
  recenter: (en: JamSourceEN) => void;
  setBandEnable: (src: number, band: number, on: boolean) => void;
  /** 壳显示总透明度（0..1，乘进材质 opacityNode，不触发重算） */
  setOpacity: (v: number) => void;
  sourceCount: number;
  getSourceEN: (i: number) => JamSourceEN;
  /** 波前是否已收敛（收敛后 compute 停算，仅保留渲染） */
  converged: () => boolean;
  /** 当前合成完成的场纹理（synth ping-pong：返回最近一次写入的 A 或 B）。
   *  供屏幕空间覆盖热图采样（alpha = 0.1×clamp(score/30)）。 */
  getDisplayTexture: () => THREE.Texture;
  /** 释放本实例全部 GPU 资源（壳几何/材质/场纹理/storage buffer）。
   *  调用方负责 scene.remove(mesh)；障碍纹理由调用方（obstacle 属主）另行 dispose。 */
  dispose: () => void;
  /** debug：底层 GPU buffer attribute（用 renderer.getArrayBufferAsync 回读） */
  buffers: { meta: THREE.BufferAttribute; dist: THREE.BufferAttribute };
};

const MAX_SOURCES = 4;
const INF = 0xffffffff; // uint 最大值 = 未到达
const INF_F = 4.0e9; // f32 安全比较阈值
const DIST_SCALE = 10; // dist 以 0.1m 为单位存 uint
const FRONTIER_CAP = 1 << 21; // 2M frontier 元素上限（一源一盒：单源 2km 半径 frontier 峰值远低于此；较原 8M 全岛版省 48MB/源）
const TURN_LAMBDA = (Math.LN2 * 2) / (Math.PI / 2); // ≈0.883
const K_ROUNDS = 8; // 每帧波前松弛轮数
const SYNTH_EVERY = 2; // 每 N 帧合成一次场强
const SHELL_CAP = 1 << 21; // 壳实例上限 209 万（2026-09-14 重标定后高档场球面 1.2km 半径≈46 万 + 街谷侧壁/遮挡阴影面数倍裕量）
const SHELL_EPS = 1e-4; // 壳表面 alpha 阈值（同 Godot generate_unified_boxes.glsl 的 EPS：score>0 即画）
const LOG10 = 1 / Math.log2(10); // log10(x) = log2(x)*LOG10
const WG = 64; // expand workgroup 尺寸（convert 里 dispatch 计算必须一致）

// 26 邻域方向
const DIRS: {
  o: [number, number, number];
  u: [number, number, number];
  step: number;
}[] = [];
for (const dx of [-1, 0, 1])
  for (const dy of [-1, 0, 1])
    for (const dz of [-1, 0, 1]) {
      if (dx === 0 && dy === 0 && dz === 0) continue;
      const l = Math.hypot(dx, dy, dz);
      DIRS.push({ o: [dx, dy, dz], u: [dx / l, dy / l, dz / l], step: l });
    }

export function createJamField(
  renderer: THREE.WebGPURenderer,
  obstacles: ObstacleField,
  grid: [number, number, number],
  boxMin: THREE.Vector3,
  boxSize: THREE.Vector3,
  sources: JamSourceEN[],
  opts?: {
    /** 本实例参与合成的频段下标（jambands.BANDS 索引）；默认全部 7 段。
     *  分层架构下每层只算本层频段，跨层组合由 heatmap 取 max 完成。 */
    bands?: number[];
    /** 波前最大传播距离（米）；默认 2000。低档层需 ≥22000 覆盖 433M 自然衰减半径 */
    maxDistM?: number;
    /** 收敛预算帧数（×K_ROUNDS 轮）；默认 480。粗网格层测地线更长，需加大 */
    expandFrames?: number;
    /** 盒水平壁显示渐隐宽度（米）；默认 120 */
    fadeXZm?: number;
  },
): JamField {
  const [GX, GY, GZ] = grid;
  const CELL_COUNT = GX * GY * GZ;
  const srcCount = Math.min(sources.length, MAX_SOURCES);
  const vx = boxSize.x / GX;
  const bandIds = opts?.bands ?? BANDS.map((_, i) => i);
  const MAX_DIST_M = opts?.maxDistM ?? 2000;
  const EXPAND_FRAMES = opts?.expandFrames ?? 480;
  const FADE_XZ = opts?.fadeXZm ?? 120;

  // ---------- GPU 缓冲 ----------
  // TSL 的 storage()/atomicLoad() 类型定义比运行时严格（string 重载、AtomicFunctionNode
  // 缺比较方法链），统一走 any helper，运行时行为不受影响
  const storageAny = (attr: any, type: string, count: number): any =>
    (storage as any)(attr, type, count);
  // 关键坑：TSL 导出的 atomicLoad/atomicAdd/... 全是 atomicFunc 工厂（自带 .toStack()），
  // 返回值路径必死——StackNode 先 build → isVoid → 生成裸语句，表达式侧拿到空串报
  // "Invalid generated code"。需要返回值时必须绕过工厂：nodeProxy(AtomicFunctionNode)
  // 构造裸节点（无 StackNode parent → 生成表达式；nodeProxy 包装才有链式方法）。
  // 关键坑（A/B 实验实证，见 AGENTS.md）：atomic 取值后若只做同类型 .toUint()，
  // ConvertType 短路返回原节点 → parents 仍只有 StackNode → isVoid → 生成裸语句、值变 0。
  // 安全用法：①uint 语境直接用 atomic 节点本身（比较/索引/算术节点会注册 parent）；
  // ②float 语境用 float(atomicLoad(...))（异类型转换才创建新节点 → parent 注册）。
  const aload = (p: any): any => atomicLoad(p); // uint 语境直接用
  const aloadF = (p: any): any => float(atomicLoad(p) as any); // float 语境

  const distAttr = new THREE.StorageBufferAttribute(
    new Uint32Array(CELL_COUNT),
    1,
  );
  const distBuf = storageAny(distAttr, "uint", CELL_COUNT).toAtomic();
  const srcAttr = new THREE.StorageBufferAttribute(
    new Uint32Array(CELL_COUNT),
    1,
  );
  const srcBuf = storageAny(srcAttr, "uint", CELL_COUNT);
  const frontierAAttr = new THREE.StorageBufferAttribute(
    new Uint32Array(FRONTIER_CAP),
    1,
  );
  const frontierA = storageAny(frontierAAttr, "uint", FRONTIER_CAP);
  const frontierBAttr = new THREE.StorageBufferAttribute(
    new Uint32Array(FRONTIER_CAP),
    1,
  );
  const frontierB = storageAny(frontierBAttr, "uint", FRONTIER_CAP);
  const metaAttr = new THREE.StorageBufferAttribute(new Uint32Array(4), 1);
  const metaBuf = storageAny(metaAttr, "uint", 4).toAtomic();
  const dispAttr = new THREE.IndirectStorageBufferAttribute(
    new Uint32Array(3) as any,
    1,
  );
  const dispBuf = storageAny(dispAttr, "uint", 3).toAtomic();

  // 壳实例 buffer：emit pass 写（RW），顶点阶段只读视图（同一 GPU buffer 另一 TSL 节点）
  const shellCellAttr = new THREE.StorageBufferAttribute(
    new Uint32Array(SHELL_CAP),
    1,
  );
  const shellColorAttr = new THREE.StorageBufferAttribute(
    new Uint32Array(SHELL_CAP),
    1,
  );
  const shellCellBuf = storageAny(shellCellAttr, "uint", SHELL_CAP);
  const shellColorBuf = storageAny(shellColorAttr, "uint", SHELL_CAP);
  const shellCellRO = storageAny(shellCellAttr, "uint", SHELL_CAP).toReadOnly();
  const shellColorRO = storageAny(
    shellColorAttr,
    "uint",
    SHELL_CAP,
  ).toReadOnly();

  const dirData = new Float32Array(26 * 4);
  DIRS.forEach((d, i) =>
    dirData.set([d.u[0], d.u[1], d.u[2], d.step * vx], i * 4),
  );
  const dirAttr = new THREE.BufferAttribute(dirData, 4);
  const dirBuf = storage(dirAttr, "vec4", 26).toReadOnly();

  const srcPosAttr = new THREE.BufferAttribute(
    new Float32Array(MAX_SOURCES * 4),
    4,
  );
  const srcPosBuf = storage(srcPosAttr, "vec4", MAX_SOURCES).toReadOnly();
  const srcMaskAttr = new THREE.BufferAttribute(
    new Uint32Array(MAX_SOURCES).fill(0x7f),
    1,
  );
  const srcMaskBuf = storage(srcMaskAttr, "uint", MAX_SOURCES).toReadOnly();
  const bandAttr = new THREE.BufferAttribute(packBandBuffer(), 4);
  const bandBuf = storage(bandAttr, "vec4", BANDS.length * 2).toReadOnly();

  const createFieldTex = (name: string) => {
    const t = new THREE.Storage3DTexture(GX, GY, GZ);
    t.name = name;
    t.format = THREE.RGBAFormat;
    t.type = THREE.HalfFloatType;
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    return t;
  };
  const fieldTexA = createFieldTex("jamfield A");
  const fieldTexB = createFieldTex("jamfield B");

  // ---------- 工具 ----------
  const uBoxMin = uniform(boxMin.clone());
  const uBoxSize = uniform(boxSize.clone());
  const gridDims = vec3(GX, GY, GZ);
  // int cell → 线性索引（int；转 uint 由调用方做）
  const cellIndex = (c: any) =>
    c.z
      .mul(int(GX * GY))
      .add(c.y.mul(int(GX)))
      .add(c.x);
  const inGrid = (c: any) =>
    c.x
      .greaterThanEqual(0)
      .and(c.y.greaterThanEqual(0))
      .and(c.z.greaterThanEqual(0))
      .and(c.x.lessThan(int(GX)))
      .and(c.y.lessThan(int(GY)))
      .and(c.z.lessThan(int(GZ)));
  const solidAtCell = (c: any) =>
    texture3D(
      obstacles.texture,
      vec3(c).add(0.5).div(gridDims),
      float(0),
    ).r.greaterThan(0.5);

  // ---------- Pass: reset（初始化 / 源移动后） ----------
  const resetNode = Fn(() => {
    atomicStore(distBuf.element(instanceIndex), uint(INF));
    srcBuf.element(instanceIndex).assign(uint(INF));
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(metaBuf.element(0), uint(0));
      atomicStore(metaBuf.element(1), uint(0));
    });
  })()
    .compute(CELL_COUNT)
    .setName("jamReset");

  // ---------- Pass: seed（每源 1 线程） ----------
  const seedNode = Fn(() => {
    If(instanceIndex.lessThan(uint(srcCount)), () => {
      const p = srcPosBuf.element(instanceIndex).xyz;
      const uvw = p.sub(uBoxMin).div(uBoxSize);
      const c = vec3(
        clamp(uvw.mul(gridDims), vec3(0), gridDims.sub(1)),
      ).floor();
      const ic = vec3(int(c.x), int(c.y), int(c.z));
      const idx = uint(cellIndex(ic));
      atomicStore(distBuf.element(idx), uint(0));
      srcBuf.element(idx).assign(instanceIndex);
      // 推 frontierB：voxelId | dirId(=31 无方向)<<24 | srcId<<29
      const packed = idx
        .bitOr(uint(31).shiftLeft(uint(24)))
        .bitOr(instanceIndex.shiftLeft(uint(29)));
      const slot: any = atomicAdd(metaBuf.element(1), uint(1));
      If(slot.lessThan(uint(FRONTIER_CAP)), () => {
        frontierB.element(slot).assign(packed);
      });
    });
  })()
    .compute(MAX_SOURCES)
    .setName("jamSeed");

  // ---------- Pass: convert（单线程：B→A，写间接 dispatch 尺寸） ----------
  const convertNode = Fn(() => {
    // clamp 到 FRONTIER_CAP：meta[1] 无上限自增，不 clamp 会按残留 frontier 数据
    // 重复 expand 已处理体素 → 波前永不收敛
    const cB = aload(metaBuf.element(1)).min(uint(FRONTIER_CAP));
    atomicStore(metaBuf.element(0), cB);
    atomicStore(metaBuf.element(1), uint(0));
    atomicStore(
      dispBuf.element(0),
      cB
        .add(uint(WG - 1))
        .div(uint(WG))
        .max(uint(1)),
    );
    atomicStore(dispBuf.element(1), uint(1));
    atomicStore(dispBuf.element(2), uint(1));
  })()
    .compute(1)
    .setName("jamConvert");

  // ---------- Pass: expand（间接 dispatch，frontier 元素 → 26 邻居） ----------
  const turnLambda = uniform(TURN_LAMBDA);
  // frontier 数据双缓冲 ping-pong：读 readBuf、推 writeBuf（convert 只转计数）
  const makeExpand = (readBuf: any, writeBuf: any, name: string) =>
    Fn(() => {
      const countA = aload(metaBuf.element(0));
      If(instanceIndex.lessThan(countA), () => {
        const packed = readBuf.element(instanceIndex);
        const voxelId = packed.bitAnd(uint(0x00ffffff));
        const myDir = packed.bitAnd(uint(0x1f000000)).shiftRight(uint(24));
        const mySrc = packed.shiftRight(uint(29));
        const vid = int(voxelId);
        const cz = vid.div(int(GX * GY));
        const rem = vid.sub(cz.mul(int(GX * GY)));
        const cy = rem.div(int(GX));
        const cx = rem.sub(cy.mul(int(GX)));
        const d0 = aloadF(distBuf.element(voxelId));
        const maxD = srcPosBuf.element(mySrc).w.mul(DIST_SCALE);

        // 预采样 6 个面邻居固体标志（防穿墙角复用）
        const spx = solidAtCell(vec3(cx.add(1), cy, cz));
        const snx = solidAtCell(vec3(cx.sub(1), cy, cz));
        const spy = solidAtCell(vec3(cx, cy.add(1), cz));
        const sny = solidAtCell(vec3(cx, cy.sub(1), cz));
        const spz = solidAtCell(vec3(cx, cy, cz.add(1)));
        const snz = solidAtCell(vec3(cx, cy, cz.sub(1)));

        DIRS.forEach((d, dirId) => {
          const nb = vec3(cx.add(d.o[0]), cy.add(d.o[1]), cz.add(d.o[2]));
          If(inGrid(nb), () => {
            If(solidAtCell(nb).not(), () => {
              // 防穿墙角：对角移动要求所有非零分量的面邻居全非固体
              let open: any = null;
              if (d.o[0] !== 0) open = d.o[0] > 0 ? spx.not() : snx.not();
              if (d.o[1] !== 0) {
                const f = d.o[1] > 0 ? spy.not() : sny.not();
                open = open ? open.and(f) : f;
              }
              if (d.o[2] !== 0) {
                const f = d.o[2] > 0 ? spz.not() : snz.not();
                open = open ? open.and(f) : f;
              }
              If(open, () => {
                // 转向惩罚：exp(λ·acos(dirPrev·dirNew))；种子首步（dir=31）免罚
                const pen = float(1).toVar();
                If(myDir.notEqual(uint(31)), () => {
                  const myDirV = dirBuf.element(myDir.min(uint(25))).xyz;
                  const offU = dirBuf.element(uint(dirId)).xyz;
                  const cosA = clamp(dot(myDirV, offU), -1, 1);
                  pen.assign(exp(turnLambda.mul(acos(cosA))));
                });
                const step = dirBuf.element(uint(dirId)).w.mul(DIST_SCALE);
                const nd = d0.add(step.mul(pen));
                If(nd.lessThan(maxD).and(nd.lessThan(INF_F)), () => {
                  const ni = uint(cellIndex(nb));
                  const old: any = atomicMin(distBuf.element(ni), uint(nd));
                  If(float(old).greaterThan(nd), () => {
                    srcBuf.element(ni).assign(mySrc);
                    const pk = ni
                      .bitOr(uint(dirId).shiftLeft(uint(24)))
                      .bitOr(mySrc.shiftLeft(uint(29)));
                    const slot: any = atomicAdd(metaBuf.element(1), uint(1));
                    If(slot.lessThan(uint(FRONTIER_CAP)), () => {
                      writeBuf.element(slot).assign(pk);
                    });
                  });
                });
              });
            });
          });
        });
      });
    })()
      .compute(dispAttr as any)
      .setName(name);
  const expandBA = makeExpand(frontierB, frontierA, "jamExpandBA"); // 读 B 推 A
  const expandAB = makeExpand(frontierA, frontierB, "jamExpandAB"); // 读 A 推 B

  // ---------- Pass: synth（场强合成 + 时间平滑，ping-pong） ----------
  const uDtS = uniform(0.033);
  const makeSynth = (
    writeTex: THREE.Storage3DTexture,
    prevTex: THREE.Storage3DTexture,
    name: string,
  ) =>
    Fn(() => {
      const coord = vec3(
        int(instanceIndex.mod(uint(GX))),
        int(instanceIndex.div(uint(GX)).mod(uint(GY))),
        int(instanceIndex.div(uint(GX * GY))),
      );
      const uvw = coord.add(0.5).div(gridDims);
      const cellPos = uBoxMin.add(uvw.mul(uBoxSize));
      const idx = instanceIndex;
      const d = aloadF(distBuf.element(idx));
      const prevV = texture3D(prevTex, uvw, float(0));
      const outV = vec4(0, 0, 0, 0).toVar(); // a 必须初始 0：未到达体素(INF)若 a=1 会成吸光黑洞
      // DEBUG(JS 生成期开关): 直接可视化等效距离场
      // r=dist/3km, g=未到达(INF)标记, b=源id/4
      const DBG_DIST = false;
      if (DBG_DIST) {
        // r=dist/3km, g=INF 标记(smoothstep 避开 f32 精度), b=frontier 计数/500
        outV.assign(
          vec4(
            clamp(d.div(7000), 0, 1),
            smoothstep(float(3.9e9), float(4.29e9), d),
            clamp(aloadF(metaBuf.element(0)).div(500), 0, 1),
            1,
          ),
        );
      }
      If(
        d
          .lessThan(INF_F)
          .and(
            (DBG_DIST
              ? float(0).greaterThan(0.5)
              : float(1).greaterThan(0.5)) as any,
          ),
        () => {
          const si = srcBuf.element(idx).min(uint(MAX_SOURCES - 1));
          const sp = srcPosBuf.element(si);
          const toR = cellPos.sub(sp.xyz);
          const distM = max(length(toR), 1.0);
          const effKm = d.div(DIST_SCALE).mul(0.001).max(1e-4);
          const horiz = length(vec3(toR.x, 0, toR.z));
          const cosT = clamp(horiz.div(distM), 1e-3, 1);
          const gt = float(GMAX_DBI).add(
            float(10 * VERTICAL_N).mul(log2(cosT).mul(LOG10)),
          );
          const mask = srcMaskBuf.element(si);
          // Godot 原版 场强→alpha→选段（compute_unified_field.glsl 完整迁移）：
          //   pr = pt + gt − lfs − MARGIN（原版的 wall_loss/ldiff 两项不迁——
          //   本实现用 DPM 等效距离 d 替代直线距离，障碍阻挡/转向惩罚已折算进 d，
          //   再计会双重扣减）；每频段 alpha_i = 0.1×clamp(score_i/30)；
          // 单频段选择（同 field pass 尾部）：低频→高频遍历，a≥当前最优且 a>1e-4
          // 则更新（平局取高频）——近源全频段饱和(0.1)由最高频胜出=红芯，
          // 走远后高频先掉出饱和、下一频段接管（橙→黄→绿→青→蓝），
          // 远郊只剩 433 未饱和，深蓝渐淡至 score=0。
          const bestA = float(0).toVar();
          const bestColor = vec3(0, 0, 0.498).toVar(); // 兜底 433 深蓝（a=0 时不可见）
          for (const bi of bandIds) {
            // 分层：本实例只合成本层频段，跨层 max 由 heatmap 完成
            If(mask.bitAnd(uint(1 << bi)).notEqual(uint(0)), () => {
              const freq = bandBuf.element(bi * 2).w;
              const pt2 = bandBuf.element(bi * 2 + 1).x;
              const lfs = float(20 * LOG10)
                .mul(log2(effKm))
                .add(float(20 * LOG10).mul(log2(freq)))
                .add(32.45);
              const pr = pt2.add(gt).sub(lfs).sub(float(MARGIN_DBM));
              const sc = pr.sub(bandBuf.element(bi * 2 + 1).y);
              const a = clamp(sc.div(30), 0, 1).mul(0.1);
              If(a.greaterThan(1e-4).and(a.greaterThanEqual(bestA)), () => {
                bestA.assign(a);
                bestColor.assign(bandBuf.element(bi * 2).rgb);
              });
            });
          }
          const avgColor = bestColor;
          // 盒边显示渐隐（只改显示纹理，不动 dist 传播）：水平渐隐宽度按层给
          // （细层 120m / 中 300m / 粗 600m），垂直渐隐带绝对锚定盒顶
          // [顶-100m, 顶-10m]——比例带会误伤粗档的山体遮挡（如海拔 933m 山峰）。
          const wallDist = min(
            min(
              cellPos.x.sub(uBoxMin.x),
              uBoxMin.x.add(uBoxSize.x).sub(cellPos.x),
            ),
            min(
              cellPos.z.sub(uBoxMin.z),
              uBoxMin.z.add(uBoxSize.z).sub(cellPos.z),
            ),
          );
          const yTop = uBoxMin.y.add(uBoxSize.y);
          const edgeFade = smoothstep(float(0), float(FADE_XZ), wallDist).mul(
            float(1).sub(smoothstep(yTop.sub(100), yTop.sub(10), cellPos.y)),
          );
          const alpha = bestA.mul(edgeFade);
          // 时间平滑（指数收敛 ~0.5s，源移动后不闪变）
          const k = float(1).sub(exp(uDtS.div(-0.5)));
          outV.assign(
            vec4(mix(prevV.rgb, avgColor, k), mix(prevV.a, alpha, k)),
          );
        },
      );
      textureStore(writeTex, coord, outV).toWriteOnly();
    })()
      .compute(CELL_COUNT)
      .setName(name);
  const synthA2B = makeSynth(fieldTexB, fieldTexA, "jamSynthAB");
  const synthB2A = makeSynth(fieldTexA, fieldTexB, "jamSynthBA");

  // ---------- Pass: shellClear（清壳实例颜色 + 计数器 meta[2]） ----------
  // 只清颜色（alpha=0 即不可见，cell 残留无所谓）；绘制按 SHELL_CAP 满量提交
  const shellClearNode = Fn(() => {
    shellColorBuf.element(instanceIndex).assign(uint(0));
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(metaBuf.element(2), uint(0));
    });
  })()
    .compute(SHELL_CAP)
    .setName("jamShellClear");

  // ---------- Pass: shellEmit（表面格检测 + append 实例，读 synth 刚写的纹理） ----------
  // 表面 = 场 alpha>EPS 且 6 面**盒内**邻居至少一个 alpha≤EPS。
  // 出界邻居不算空（2026-09-11 修复）：场半径大于盒半径时波前顶到盒壁，
  // 若出界算空会把盒四壁+顶盖整体判成表面画成实心蓝墙；配合 synth 的
  // 盒边渐隐，边界处 alpha<EPS 自然消隐，无需盒壁闭合。
  // （判定逻辑 otherwise 同 Godot generate_unified_boxes.glsl 的 is_band_surface）
  const makeShellEmit = (readTex: THREE.Storage3DTexture, name: string) =>
    Fn(() => {
      const coord = vec3(
        int(instanceIndex.mod(uint(GX))),
        int(instanceIndex.div(uint(GX)).mod(uint(GY))),
        int(instanceIndex.div(uint(GX * GY))),
      );
      const uvw = coord.add(0.5).div(gridDims);
      const f = texture3D(readTex, uvw, float(0));
      If(f.a.greaterThan(float(SHELL_EPS)), () => {
        const surface = float(0).toVar();
        const OFFS: [number, number, number][] = [
          [1, 0, 0],
          [-1, 0, 0],
          [0, 1, 0],
          [0, -1, 0],
          [0, 0, 1],
          [0, 0, -1],
        ];
        OFFS.forEach(([ox, oy, oz]) => {
          const nb = coord.add(vec3(ox, oy, oz));
          const na = texture3D(readTex, nb.add(0.5).div(gridDims), float(0)).a;
          const empty = inGrid(nb).and(na.lessThanEqual(float(SHELL_EPS)));
          surface.assign(max(surface, float(empty)));
        });
        If(surface.greaterThan(0.5), () => {
          const slot: any = atomicAdd(metaBuf.element(2), uint(1));
          If(slot.lessThan(uint(SHELL_CAP)), () => {
            shellCellBuf.element(slot).assign(instanceIndex);
            const r = clamp(f.r, 0, 1).mul(255);
            const g = clamp(f.g, 0, 1).mul(255);
            const b = clamp(f.b, 0, 1).mul(255);
            const a = clamp(f.a, 0, 1).mul(255);
            shellColorBuf.element(slot).assign(
              uint(r)
                .bitOr(uint(g).shiftLeft(uint(8)))
                .bitOr(uint(b).shiftLeft(uint(16)))
                .bitOr(uint(a).shiftLeft(uint(24))),
            );
          });
        });
      });
    })()
      .compute(CELL_COUNT)
      .setName(name);
  const shellEmitA = makeShellEmit(fieldTexA, "jamShellEmitA"); // synthB2A 后读 A
  const shellEmitB = makeShellEmit(fieldTexB, "jamShellEmitB"); // synthA2B 后读 B

  // ---------- 渲染：实例化方块壳（普通半透明 + depthTest，无 raymarch） ----------
  const shellGeom = new THREE.InstancedBufferGeometry();
  const boxGeom = new THREE.BoxGeometry(1, 1, 1);
  shellGeom.setIndex(boxGeom.getIndex());
  shellGeom.setAttribute("position", boxGeom.getAttribute("position"));
  shellGeom.instanceCount = SHELL_CAP;
  const uVoxel = uniform(
    new THREE.Vector3(boxSize.x / GX, boxSize.y / GY, boxSize.z / GZ),
  );

  const shellCellU = shellCellRO.element(instanceIndex);
  const scx = shellCellU.mod(uint(GX));
  const scy = shellCellU.div(uint(GX)).mod(uint(GY));
  const scz = shellCellU.div(uint(GX * GY));
  const cellCenter = uBoxMin.add(
    vec3(
      float(scx).add(0.5).mul(uVoxel.x),
      float(scy).add(0.5).mul(uVoxel.y),
      float(scz).add(0.5).mul(uVoxel.z),
    ),
  );
  // 空槽（color==0）坍缩到远处地下：零面积三角形直接被裁，alpha 残留不可见
  const shellPos = Fn(() => {
    const p = positionGeometry.mul(uVoxel).add(cellCenter).toVar();
    If(shellColorRO.element(instanceIndex).equal(uint(0)), () => {
      p.assign(vec3(0, -1e6, 0));
    });
    return p;
  })();

  const packedCol = shellColorRO.element(instanceIndex);
  const vCol = varying(
    vec4(
      float(packedCol.bitAnd(uint(0xff))).div(255),
      float(packedCol.bitAnd(uint(0xff00)).shiftRight(uint(8))).div(255),
      float(packedCol.bitAnd(uint(0xff0000)).shiftRight(uint(16))).div(255),
      float(packedCol.shiftRight(uint(24))).div(255),
    ),
  );

  const material = new THREE.NodeMaterial();
  material.positionNode = shellPos;
  material.colorNode = vCol.rgb;
  const uOpacity = uniform(1.0); // 面板总透明度（UI 直调，不走 synth）
  material.opacityNode = vCol.a.mul(uOpacity);
  material.transparent = true;
  material.depthWrite = false; // 壳片之间不互相挡；建筑遮挡靠 depthTest（默认开）
  material.side = THREE.DoubleSide; // 从壳内向外看也能看到对面壳

  const mesh = new THREE.Mesh(shellGeom, material);
  mesh.name = "jam-field";
  mesh.frustumCulled = false; // 实例遍布全盒，基础几何包围球无意义

  // ---------- 源管理 ----------
  const srcs = sources.slice(0, MAX_SOURCES).map((s) => ({ ...s }));
  const pushSources = () => {
    srcs.forEach((s, i) => {
      const y = obstacles.surfaceY(s.x, s.z) + 4;
      srcPosAttr.array.set([s.x, y, s.z, MAX_DIST_M], i * 4); // maxDist（米，按层档位给）
    });
    (srcPosAttr as any).needsUpdate = true;
  };
  pushSources();

  // ---------- 帧更新 ----------
  let frame = 0;
  let parity = 0;
  let needReset = true;
  // 收敛停算（固定预算制，杜绝 GPU 回读）：波前按最坏情形预算跑满即视为收敛——
  // 26 连通网格最长测地线 ≈ GX+GY+GZ 步，建筑/山体绕行给 ~5.5× 裕量（expandFrames
  // 由调用方按网格尺度给：细层 480、中层 700、粗层 560，均 ×K_ROUNDS 轮）。
  // 收敛后停 expand/convert，
  // 再补 SETTLE_SYNTHS 次 synth 让指数时间平滑到位，之后 compute 全停，
  // 壳实例 buffer 保持最后一帧内容继续画（几何静止，每帧零 compute）。
  // 不用 getArrayBufferAsync 回读 meta 判收敛：周期性 mapAsync 与 compute 并发
  // 会把 device 打挂（headless 复现：jam 启动即 LOST "Device was destroyed"）。
  // 空 frontier 时 expand 每轮只剩 64 空转线程（disp 下限 1 + instanceIndex<countA
  // 守卫），所以预算制多跑的几百轮成本可忽略。
  const SETTLE_SYNTHS = 40; // 收敛后再跑 40 次 synth（≈1.3s），让指数平滑到位
  let expandFrame = 0;
  let settleCount = 0;
  let waveDone = false;
  const run = (node: any) =>
    renderer.computeAsync(node).catch((e) => console.error("[jam]", e));

  const update = async (dt: number, _camera: THREE.Camera) => {
    frame++;
    if (needReset) {
      await run(resetNode);
      await run(seedNode);
      needReset = false;
      expandFrame = 0;
      settleCount = 0;
      waveDone = false;
    }
    if (!waveDone) {
      for (let k = 0; k < K_ROUNDS; k++) {
        await run(convertNode);
        await run(k % 2 === 0 ? expandBA : expandAB); // seed 推进 frontierB，首轮读 B
      }
      if (++expandFrame >= EXPAND_FRAMES) waveDone = true;
    }
    const synthDone = waveDone && settleCount >= SETTLE_SYNTHS;
    if (!synthDone && frame % SYNTH_EVERY === 0) {
      uDtS.value = dt * SYNTH_EVERY;
      await run(shellClearNode);
      if (parity === 0) {
        await run(synthA2B);
        await run(shellEmitB);
      } else {
        await run(synthB2A);
        await run(shellEmitA);
      }
      parity ^= 1;
      if (waveDone) settleCount++;
    }
  };

  const moveSource = (i: number, en: JamSourceEN) => {
    if (i >= srcs.length) return;
    srcs[i] = { ...en };
    pushSources();
    needReset = true; // dist 场只减不增，源移动必须重洪泛（下一帧执行）
  };
  const recenter = (en: JamSourceEN) => {
    // 只平移 x/z：盒底/盒顶海拔锚定不变（y 由建页时的盒原点决定）
    uBoxMin.value.set(
      en.x - boxSize.x / 2,
      uBoxMin.value.y,
      en.z - boxSize.z / 2,
    );
    pushSources(); // 新区域地形/楼顶高度不同，重算源发射高度
    needReset = true; // dist 场按新盒框架重洪泛
  };
  const setBandEnable = (src: number, band: number, on: boolean) => {
    if (src >= srcs.length || band >= BANDS.length) return;
    const m = srcMaskAttr.array[src] as number;
    (srcMaskAttr.array as Uint32Array)[src] = on
      ? m | (1 << band)
      : m & ~(1 << band);
    (srcMaskAttr as any).needsUpdate = true;
    // dist 传播与频段开关无关，只需补跑 SETTLE_SYNTHS 次 synth 刷新显示纹理+壳
    settleCount = 0;
  };
  const getSourceEN = (i: number) => ({ ...srcs[i] });

  const dispose = () => {
    shellGeom.dispose();
    material.dispose();
    fieldTexA.dispose();
    fieldTexB.dispose();
    // standalone storage attribute 不挂 geometry，需走后端 attributes 表单独删除
    const attrs = (
      renderer as unknown as { _attributes?: { delete: (a: unknown) => void } }
    )._attributes;
    for (const a of [
      distAttr,
      srcAttr,
      frontierAAttr,
      frontierBAttr,
      metaAttr,
      dispAttr,
      shellCellAttr,
      shellColorAttr,
      dirAttr,
      srcPosAttr,
      srcMaskAttr,
      bandAttr,
    ]) {
      try {
        attrs?.delete(a);
      } catch (e) {
        console.warn("[jam] dispose attr failed", e);
      }
    }
  };

  return {
    mesh,
    update,
    moveSource,
    recenter,
    setBandEnable,
    sourceCount: srcCount,
    getSourceEN,
    setOpacity: (v: number) => {
      uOpacity.value = v;
    },
    converged: () => waveDone && settleCount >= SETTLE_SYNTHS,
    // update 写完即翻转 parity：parity===1 时最近写的是 B，===0 时最近写的是 A
    getDisplayTexture: () => (parity === 1 ? fieldTexB : fieldTexA),
    dispose,
    buffers: { meta: metaAttr, dist: distAttr },
  };
}
