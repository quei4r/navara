// 干扰频段参数：频段表移植自 Godot 项目 scripts/helpers/gpu_voxel_band_params.gd
// （该表逐字对应空御科技"驱鹰"SD-XYQY-Q 全向全频段干扰设备官方规格书：
// 433/800/900M、1.2/1.5/2.4/5.8G 七频段可选配，干扰介入距离 2km/3km 两档）。
// 只保留干扰场合成需要的字段（Radar 频段是二值检测，无传播公式，不移植）

export type BandDef = {
  name: string;
  freqMHz: number;
  color: [number, number, number];
  rangeKm: number; // score=0 显示半径 = 有效介入距离（致死/劣化边界）
  thresholdDbm: number; // 由 rangeKm 反推：使 pr(range)=threshold
  txPower2kmDbm: number; // 按 2km 介入半径标定的发射功率
};

export const GMAX_DBI = 2.0;
export const VERTICAL_N = 4.0;
export const MARGIN_DBM = 10.0;
export const NOISE_FLOOR_DBM = -75.0;
export const VISIBLE_OFFSET_DBM = 5.0;

// [name, freqMHz, color, score=0 半径 km]
const BAND_DEFS: [string, number, [number, number, number], number][] = [
  ["433MHz", 433.0, [0.0, 0.0, 0.498], 2.5],
  ["800MHz", 800.0, [0.0, 0.157, 1.0], 2.3],
  ["900MHz", 900.0, [0.0, 0.831, 1.0], 2.3],
  ["1.2GHz", 1200.0, [0.486, 1.0, 0.475], 2.0],
  ["1.5GHz", 1500.0, [1.0, 0.898, 0.0], 1.8],
  ["2.4GHz", 2400.0, [1.0, 0.275, 0.0], 1.8],
  ["5.8GHz", 5800.0, [0.498, 0.0, 0.0], 1.2],
];

export const BAND_COUNT = BAND_DEFS.length; // 7

// 频段→档位映射（2026-09-11 拍板分层：精细度跟着频段走，算力显存放在该放的地方）：
// 低档 433/800/900M——波长远大于建筑尺度，白膜遮挡可省（杂波损耗已在 MARGIN 统计
// 吸收），只要地形山体遮挡 → 大盒粗体素纯地形传播；中档 1.2/1.5/2.4G——过渡区，
// 中盒中体素+大体量建筑；高档 5.8G——波长 5cm，建筑尺度交互主导 → 细盒细体素全遮挡。
// 同一档内频段共享同一 dist 传播场（波前几何频段无关，频段差异只在 synth 的
// pt/lfs/MARGIN），所以 7 频段只需 3 个传播实例。
export const BAND_TIERS: number[][] = [[0, 1, 2], [3, 4, 5], [6]];

// 标定半径：Godot 原版 2km 档（对应 SD-XYQY-Q 规格书"干扰介入距离 2km"；
// 原版另有 3km 档对应规格的 3km 版本，本场景 maxDist 量级用不到，只迁 2km 档）
export const JAM_RANGE_KM = 2.0;

// score=0 半径语义重标定（2026-09-14 拍板，取代 2026-09-11 的 Godot 灵敏度阈值迁移）：
// Godot 原版 threshold 是"接收灵敏度阈值"（−90…−60dBm），显示的是可检测轮廓——
// 433M 会画出 20km 大圈，远超战术上有意义的范围。现改为**有效介入距离**语义：
// score=0 轮廓 = 干扰开始压制链路的边界（致死/劣化边界）。逐频段锚点：
//   1.5G/2.4G = 1.8km、5.8G = 1.2km（上海政采同类空域反制设备验收指标）；
//   433/800/900M = 2.5/2.3km（长距跳频 RC 链路，按 Q 型整机 2-3km 规格 + 低频
//   传播优势取）；1.2G = 2.0km（整机规格）。
// 发射功率标定不变：pt = NOISE_FLOOR + VISIBLE_OFFSET − GMAX + lfs(2km) + MARGIN，
// 含 20·log(f) 项 → 全频段 pr 与频率无关，pr(2km)=NF+VO=−70dBm。
// 阈值由半径反推：pr(r)=−70−20·log10(r/2km)，令 threshold=pr(range) 即可。
// 物理效果：彩虹选段压缩到源心 ~130m 内（各频段饱和半径相近），之外由最低阈值
// 频段（433M）深蓝收尾——这正是"各频段真实覆盖差异不大"的现实图景。
export const BANDS: BandDef[] = BAND_DEFS.map(
  ([name, freqMHz, color, rangeKm]) => ({
    name,
    freqMHz,
    color,
    rangeKm,
    thresholdDbm:
      NOISE_FLOOR_DBM +
      VISIBLE_OFFSET_DBM -
      20 * Math.log10(rangeKm / JAM_RANGE_KM),
    txPower2kmDbm:
      NOISE_FLOOR_DBM +
      VISIBLE_OFFSET_DBM -
      GMAX_DBI +
      (20 * Math.log10(JAM_RANGE_KM) + 20 * Math.log10(freqMHz) + 32.45) +
      MARGIN_DBM,
  }),
);

/** 打包成 GPU readonly buffer：每频段 2×vec4（color.rgb+freq / pt2+threshold+pad） */
export function packBandBuffer(): Float32Array {
  const out = new Float32Array(BAND_COUNT * 8);
  for (let i = 0; i < BAND_COUNT; i++) {
    const b = BANDS[i];
    out[i * 8 + 0] = b.color[0];
    out[i * 8 + 1] = b.color[1];
    out[i * 8 + 2] = b.color[2];
    out[i * 8 + 3] = b.freqMHz;
    out[i * 8 + 4] = b.txPower2kmDbm;
    out[i * 8 + 5] = b.thresholdDbm;
  }
  return out;
}

/** 渲染用 7×1 调色板纹理数据（RGBA8） */
export function makePaletteData(): Uint8Array {
  const out = new Uint8Array(BAND_COUNT * 4);
  for (let i = 0; i < BAND_COUNT; i++) {
    const c = BANDS[i].color;
    out[i * 4 + 0] = Math.round(c[0] * 255);
    out[i * 4 + 1] = Math.round(c[1] * 255);
    out[i * 4 + 2] = Math.round(c[2] * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}
