// 白膜建筑 footprint 解码：gba-server 的 MVT 瓦片 -> 建筑 footprint 多边形列表
// （r185 buildings.ts 只搬 decodeBuildingPolys——体素化只需 footprint，
//   白膜 mesh 构建部分不搬；lngLatToENU 改为按源参数化的 LocalCoords）
import { VectorTile } from "@mapbox/vector-tile";
import { PbfReader } from "pbf";
import * as THREE from "three/webgpu";

import { tileBounds, type LocalCoords } from "./coords";
import { sampleHeight, type HeightField } from "./heightfield";

const MVT_URL = (z: number, x: number, y: number) =>
  `http://localhost:8787/tiles/${z}/${x}/${y}.mvt`;

type RingEN = {
  pts: THREE.Vector2[]; // (东, 北) 本地米
  lnglats: { lng: number; lat: number }[];
};

export type BuildingPoly = {
  outer: THREE.Vector2[]; // (东,北)，逆时针（three Shape 规范）
  holes: THREE.Vector2[][]; // (东,北)，顺时针
  ground: number; // 基底地面高程（米）
  top: number; // 楼顶绝对高程（米）
};

/** 单块 MVT 瓦片 -> 建筑 footprint 多边形列表。瓦片不存在（404）或无建筑时返回 null。 */
export async function decodeBuildingPolys(
  coords: LocalCoords,
  x: number,
  y: number,
  z: number,
  hf: HeightField,
): Promise<BuildingPoly[] | null> {
  const resp = await fetch(MVT_URL(z, x, y));
  if (resp.status === 404) return null;
  if (!resp.ok) throw new Error(`mvt ${z}/${x}/${y}: HTTP ${resp.status}`);
  const tile = new VectorTile(
    new PbfReader(new Uint8Array(await resp.arrayBuffer())),
  );
  const layer = tile.layers["building"];
  if (!layer || layer.length === 0) return null;

  const extent = layer.extent || 4096;
  const b = tileBounds(x, y, z);
  const toLngLat = (gx: number, gy: number) => ({
    lng: b.west + (gx / extent) * (b.east - b.west),
    lat: b.north - (gy / extent) * (b.north - b.south),
  });

  const polys: BuildingPoly[] = [];

  for (let i = 0; i < layer.length; i++) {
    const feature = layer.feature(i);
    const rawH = Number(feature.properties?.height);
    const height = Math.min(
      Math.max(Number.isFinite(rawH) && rawH > 0 ? rawH : 6, 2),
      600,
    );

    const rings: RingEN[] = [];
    for (const ring of feature.loadGeometry()) {
      const pts: THREE.Vector2[] = [];
      const lnglats: { lng: number; lat: number }[] = [];
      const nPts =
        ring.length > 1 && ring[0].equals(ring[ring.length - 1])
          ? ring.length - 1
          : ring.length;
      for (let k = 0; k < nPts; k++) {
        const ll = toLngLat(ring[k].x, ring[k].y);
        const en = coords.lngLatToENU(ll.lng, ll.lat);
        pts.push(new THREE.Vector2(en.x, -en.z)); // (东, 北)：z 南 => 北 = -z
        lnglats.push(ll);
      }
      if (pts.length >= 3) rings.push({ pts, lnglats });
    }
    if (rings.length === 0) continue;

    // 分类外环/洞：MVT 规范中外环在 y-down 瓦片坐标为顺时针，
    // 转到 (东,北) y 翻转后变为顺时针（isClockWise=true），洞相反。
    //（实测 gba-tiler 数据符合该规范，勿按常规 y-up 直觉取反）
    const grouped: { outer: RingEN; holes: RingEN[] }[] = [];
    for (const ring of rings) {
      if (!THREE.ShapeUtils.isClockWise(ring.pts)) {
        const poly = grouped[grouped.length - 1];
        if (poly) poly.holes.push(ring);
      } else {
        grouped.push({ outer: ring, holes: [] });
      }
    }

    for (const poly of grouped) {
      // three Shape 规范：外环逆时针、洞顺时针
      const outerPts = THREE.ShapeUtils.isClockWise(poly.outer.pts)
        ? [...poly.outer.pts].reverse()
        : poly.outer.pts;
      const holePts = poly.holes.map((h) =>
        THREE.ShapeUtils.isClockWise(h.pts) ? h.pts : [...h.pts].reverse(),
      );

      // 用外环质心采样地面高程
      let clng = 0;
      let clat = 0;
      for (const ll of poly.outer.lnglats) {
        clng += ll.lng;
        clat += ll.lat;
      }
      clng /= poly.outer.lnglats.length;
      clat /= poly.outer.lnglats.length;
      const ground = sampleHeight(hf, clng, clat);

      polys.push({
        outer: outerPts,
        holes: holePts,
        ground,
        top: ground + height,
      });
    }
  }

  return polys.length > 0 ? polys : null;
}
