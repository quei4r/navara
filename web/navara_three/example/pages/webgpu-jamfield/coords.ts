// 坐标工具：Web Mercator 瓦片编号 <-> 经纬度 <-> 本地 ENU 米制坐标
// 本地坐标系：每个干扰源一个 ENU 原点（x=东，y=上，z=南，米，符合 three.js y-up 习惯），
// 由 makeLocalCoords 工厂生成（r185 原版是写死的全页 ORIGIN，Navara 版改为一源一原点）

export type LocalCoords = {
  /** 经纬度 -> 本地 ENU（x 东 / z 南，米） */
  lngLatToENU: (lng: number, lat: number) => { x: number; z: number };
  /** 本地 ENU -> 经纬度 */
  enuToLngLat: (x: number, z: number) => { lng: number; lat: number };
};

export function makeLocalCoords(
  originLng: number,
  originLat: number,
): LocalCoords {
  const latRad = (originLat * Math.PI) / 180;
  const metersPerDegLat =
    111132.954 - 559.822 * Math.cos(2 * latRad) + 1.175 * Math.cos(4 * latRad);
  const metersPerDegLng = 111132.954 * Math.cos(latRad);
  return {
    lngLatToENU: (lng, lat) => ({
      x: (lng - originLng) * metersPerDegLng,
      z: -(lat - originLat) * metersPerDegLat,
    }),
    enuToLngLat: (x, z) => ({
      lng: originLng + x / metersPerDegLng,
      lat: originLat - z / metersPerDegLat,
    }),
  };
}

/** 经纬度 -> 瓦片编号（浮点，z 层级） */
export function lngLatToTile(
  lng: number,
  lat: number,
  z: number,
): { x: number; y: number } {
  const n = 2 ** z;
  const latRad = (lat * Math.PI) / 180;
  return {
    x: ((lng + 180) / 360) * n,
    y:
      ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) *
      n,
  };
}

/** 瓦片编号 -> 该瓦片西北角经纬度 */
export function tileToLngLat(
  x: number,
  y: number,
  z: number,
): { lng: number; lat: number } {
  const n = 2 ** z;
  const lng = (x / n) * 360 - 180;
  const lat =
    (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) * 180) / Math.PI;
  return { lng, lat };
}

export type TileBounds = {
  west: number;
  east: number;
  north: number;
  south: number;
};

/** 瓦片经纬度范围 */
export function tileBounds(x: number, y: number, z: number): TileBounds {
  const nw = tileToLngLat(x, y, z);
  const se = tileToLngLat(x + 1, y + 1, z);
  return { west: nw.lng, east: se.lng, north: nw.lat, south: se.lat };
}

export type TileRange = {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  z: number;
};

/** 经纬度 bbox -> 覆盖它的整数瓦片范围 */
export function tileRangeForBBox(
  bbox: { west: number; south: number; east: number; north: number },
  z: number,
): TileRange {
  const nw = lngLatToTile(bbox.west, bbox.north, z);
  const se = lngLatToTile(bbox.east, bbox.south, z);
  return {
    x0: Math.floor(nw.x),
    x1: Math.floor(se.x),
    y0: Math.floor(nw.y),
    y1: Math.floor(se.y),
    z,
  };
}
