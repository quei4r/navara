// Terrarium PNG 高程瓦片：抓取、解码为高度场，支持按经纬度双线性采样
import { tileBounds, type TileBounds } from "./coords";

export type HeightField = {
  data: Float32Array;
  size: number; // 边长（正方形）
  bounds: TileBounds;
};

const TERRARIUM_URL = (z: number, x: number, y: number) =>
  `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;

function decodeTerrarium(rgba: Uint8ClampedArray, size: number): Float32Array {
  const out = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) {
    const o = i * 4;
    // Terrarium: h = r*256 + g + b/256 - 32768
    out[i] = rgba[o] * 256 + rgba[o + 1] + rgba[o + 2] / 256 - 32768;
  }
  return out;
}

export async function fetchHeightField(
  x: number,
  y: number,
  z: number,
): Promise<HeightField> {
  const resp = await fetch(TERRARIUM_URL(z, x, y));
  if (!resp.ok)
    throw new Error(`terrarium ${z}/${x}/${y}: HTTP ${resp.status}`);
  const bmp = await createImageBitmap(await resp.blob());
  const size = bmp.width; // 通常 256
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d", {
    willReadFrequently: true,
  }) as CanvasRenderingContext2D;
  ctx.drawImage(bmp, 0, 0);
  const img = ctx.getImageData(0, 0, size, size);
  bmp.close();
  return {
    data: decodeTerrarium(img.data, size),
    size,
    bounds: tileBounds(x, y, z),
  };
}

/** 按经纬度双线性采样高程（米）。越界时钳到边缘。 */
export function sampleHeight(
  hf: HeightField,
  lng: number,
  lat: number,
): number {
  const { west, east, north, south } = hf.bounds;
  const u = ((lng - west) / (east - west)) * (hf.size - 1);
  const v = ((north - lat) / (north - south)) * (hf.size - 1); // 行 0 在北边
  const cx = Math.min(Math.max(u, 0), hf.size - 1);
  const cy = Math.min(Math.max(v, 0), hf.size - 1);
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, hf.size - 1);
  const y1 = Math.min(y0 + 1, hf.size - 1);
  const fx = cx - x0;
  const fy = cy - y0;
  const d = hf.data;
  const s = hf.size;
  const h00 = d[y0 * s + x0];
  const h10 = d[y0 * s + x1];
  const h01 = d[y1 * s + x0];
  const h11 = d[y1 * s + x1];
  return (
    (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy
  );
}
