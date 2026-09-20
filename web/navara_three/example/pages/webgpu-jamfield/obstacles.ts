// 场景碰撞场：把地形高度场 + 建筑 footprint 体素化成一张障碍物 3D 纹理（R8，固体=255）
// 供干扰场 DPM 波前的固体边界条件使用；同时提供表面高度查询（发射器贴地/贴楼顶用）
// （r185 obstacles.ts 移植：enuToLngLat 改为按源参数化的 LocalCoords，其余逐字）
import * as THREE from "three/webgpu";

import { decodeBuildingPolys } from "./buildings";
import { tileRangeForBBox, type LocalCoords } from "./coords";
import {
  fetchHeightField,
  sampleHeight,
  type HeightField,
} from "./heightfield";

export type ObstacleField = {
  texture: THREE.Data3DTexture;
  /** 本地 (x,z) 处的表面高度（地形或楼顶，米）；用于发射器贴表面 */
  surfaceY: (x: number, z: number) => number;
  columns: Float32Array; // GX*GZ，每列表面高度
  stats: { tiles: number; footprints: number; solidRatio: number };
};

/** 射线法点在多边形内（环为 (东,北) 平面点） */
function pip(px: number, py: number, ring: THREE.Vector2[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i].x;
    const yi = ring[i].y;
    const xj = ring[j].x;
    const yj = ring[j].y;
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}

type BuiltData = {
  data: Uint8Array;
  colY: Float32Array;
  stats: { tiles: number; footprints: number; solidRatio: number };
};

/** 抓取 + 体素化主体：以 center/worldSize 定义的体积盒（y 轴向上）为范围。
 *  opts.buildings=false 时跳过建筑 footprint（低频档：波长远大于建筑尺度，
 *  只保留地形山体遮挡，省掉 MVT 抓取+栅格化）。 */
async function buildData(
  coords: LocalCoords,
  center: THREE.Vector3,
  worldSize: [number, number, number],
  grid: [number, number, number],
  tileZ: number,
  opts?: { buildings?: boolean },
): Promise<BuiltData> {
  const withBuildings = opts?.buildings !== false;
  const t0 = performance.now();
  const [SX, SY, SZ] = worldSize;
  const [GX, GY, GZ] = grid;
  const bottom = center.y - SY / 2;

  // 覆盖范围的经纬度 bbox -> z15 瓦片（本地坐标 z 向南，故 -SZ/2 一侧是北）
  const cNorth = coords.enuToLngLat(center.x, center.z - SZ / 2);
  const cSouth = coords.enuToLngLat(center.x, center.z + SZ / 2);
  const cWest = coords.enuToLngLat(center.x - SX / 2, center.z);
  const cEast = coords.enuToLngLat(center.x + SX / 2, center.z);
  const range = tileRangeForBBox(
    { west: cWest.lng, east: cEast.lng, north: cNorth.lat, south: cSouth.lat },
    tileZ,
  );

  // 逐瓦片抓高度场与建筑 footprint
  type TileData = {
    hf: HeightField;
    polys: { outer: THREE.Vector2[]; holes: THREE.Vector2[][]; top: number }[];
  };
  const tiles: TileData[] = [];
  let footprints = 0;
  for (let x = range.x0; x <= range.x1; x++) {
    for (let y = range.y0; y <= range.y1; y++) {
      let hf: HeightField;
      try {
        hf = await fetchHeightField(x, y, tileZ);
      } catch (e) {
        console.warn(`[obstacle] heightfield ${x}/${y} failed`, e);
        continue;
      }
      let polys: TileData["polys"] = [];
      if (withBuildings) {
        try {
          polys = (await decodeBuildingPolys(coords, x, y, tileZ, hf)) ?? [];
        } catch (e) {
          console.warn(`[obstacle] mvt ${x}/${y} failed`, e);
        }
      }
      footprints += polys.length;
      tiles.push({ hf, polys });
    }
  }

  // 每列（x,z）表面高度：先填地形（海面抬到 0）
  const colY = new Float32Array(GX * GZ);
  const colWorld = (ix: number, iz: number) => ({
    x: center.x + ((ix + 0.5) / GX - 0.5) * SX,
    z: center.z + ((iz + 0.5) / GZ - 0.5) * SZ,
  });
  for (let iz = 0; iz < GZ; iz++) {
    for (let ix = 0; ix < GX; ix++) {
      const w = colWorld(ix, iz);
      const ll = coords.enuToLngLat(w.x, w.z);
      let h = 0;
      for (const t of tiles) {
        const b = t.hf.bounds;
        if (
          ll.lng >= b.west &&
          ll.lng <= b.east &&
          ll.lat >= b.south &&
          ll.lat <= b.north
        ) {
          h = sampleHeight(t.hf, ll.lng, ll.lat);
          break;
        }
      }
      colY[iz * GX + ix] = Math.max(h, 0);
    }
  }

  // 建筑 footprint 栅格化：按 footprint bbox 反查列范围再 pip（避免全网格×全建筑爆炸）
  for (const t of tiles) {
    for (const p of t.polys) {
      let minE = Infinity;
      let maxE = -Infinity;
      let minN = Infinity;
      let maxN = -Infinity;
      for (const pt of p.outer) {
        if (pt.x < minE) minE = pt.x;
        if (pt.x > maxE) maxE = pt.x;
        if (pt.y < minN) minN = pt.y;
        if (pt.y > maxN) maxN = pt.y;
      }
      // (东,北) -> 列索引：E=x，N=-z
      const ix0 = Math.max(
        0,
        Math.floor(((minE - (center.x - SX / 2)) / SX) * GX),
      );
      const ix1 = Math.min(
        GX - 1,
        Math.floor(((maxE - (center.x - SX / 2)) / SX) * GX),
      );
      const iz0 = Math.max(
        0,
        Math.floor(((-maxN - (center.z - SZ / 2)) / SZ) * GZ),
      );
      const iz1 = Math.min(
        GZ - 1,
        Math.floor(((-minN - (center.z - SZ / 2)) / SZ) * GZ),
      );
      for (let iz = iz0; iz <= iz1; iz++) {
        for (let ix = ix0; ix <= ix1; ix++) {
          const w = colWorld(ix, iz);
          const pe = w.x;
          const pn = -w.z;
          if (!pip(pe, pn, p.outer)) continue;
          if (p.holes.some((h) => pip(pe, pn, h))) continue;
          const idx = iz * GX + ix;
          if (p.top > colY[idx]) colY[idx] = p.top;
        }
      }
    }
  }

  // 体素填充（x 最快、y 次之、z 最慢，与 Data3DTexture/textureStore 布局一致）
  const data = new Uint8Array(GX * GY * GZ);
  let solidCount = 0;
  for (let iz = 0; iz < GZ; iz++) {
    for (let iy = 0; iy < GY; iy++) {
      const worldY = bottom + ((iy + 0.5) / GY) * SY;
      const rowBase = GX * (iy + GY * iz);
      const colBase = iz * GX;
      for (let ix = 0; ix < GX; ix++) {
        if (worldY < colY[colBase + ix]) {
          data[rowBase + ix] = 255;
          solidCount++;
        }
      }
    }
  }

  const stats = {
    tiles: tiles.length,
    footprints,
    solidRatio: solidCount / (GX * GY * GZ),
  };
  console.log(
    `[obstacle] built in ${((performance.now() - t0) / 1000).toFixed(1)}s:`,
    JSON.stringify(stats),
  );
  return { data, colY, stats };
}

/** 表面高度查询闭包：捕获本次构建的圆心与列数据 */
function makeSurfaceY(
  center: THREE.Vector3,
  worldSize: [number, number, number],
  grid: [number, number, number],
  colY: Float32Array,
): (x: number, z: number) => number {
  const [SX, , SZ] = worldSize;
  const [GX, , GZ] = grid;
  return (x: number, z: number): number => {
    const ix = Math.min(
      GX - 1,
      Math.max(0, Math.floor(((x - (center.x - SX / 2)) / SX) * GX)),
    );
    const iz = Math.min(
      GZ - 1,
      Math.max(0, Math.floor(((z - (center.z - SZ / 2)) / SZ) * GZ)),
    );
    return colY[iz * GX + ix];
  };
}

/**
 * 以 center/worldSize 定义的体积盒（y 轴向上）为范围，
 * 把地形与建筑体素化。grid 为模拟网格分辨率。
 * （Navara 版：一源一盒原地重建（rebuild）未搬——UI 无移动交互）
 */
export async function buildObstacleField(
  coords: LocalCoords,
  center: THREE.Vector3,
  worldSize: [number, number, number],
  grid: [number, number, number],
  tileZ = 15,
  opts?: { buildings?: boolean },
): Promise<ObstacleField> {
  const [GX, GY, GZ] = grid;
  const built = await buildData(coords, center, worldSize, grid, tileZ, opts);

  const texture = new THREE.Data3DTexture(built.data, GX, GY, GZ);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;

  const field: ObstacleField = {
    texture,
    surfaceY: makeSurfaceY(center, worldSize, grid, built.colY),
    columns: built.colY,
    stats: built.stats,
  };
  return field;
}
