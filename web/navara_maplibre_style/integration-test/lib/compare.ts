import pixelmatch from "pixelmatch";
import sharp from "sharp";

type RgbaImage = { data: Buffer; width: number; height: number };

export type Comparison = {
  diffRatio: number;
  expectedPath: string;
  sizeMismatch: boolean;
  diff: RgbaImage;
};

/**
 * Flattened onto white: MapLibre's expected images leave unpainted pixels
 * transparent, which the harness paints with a white backdrop, and pixelmatch
 * would otherwise blend transparency onto its own pattern.
 */
async function readRgba(input: string | Buffer): Promise<RgbaImage> {
  const { data, info } = await sharp(input)
    .flatten({ background: "#ffffff" })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

/**
 * Diffs `actual` against every expected image the way MapLibre does
 * (pixelmatch, mismatched / total pixels) and keeps the closest one. A size
 * mismatch is resized away so the case still gets a score.
 */
export async function compareToExpected(
  actualPng: Buffer,
  expectedPaths: string[],
  threshold: number,
): Promise<Comparison> {
  const original = await readRgba(actualPng);
  let best: Comparison | undefined;
  for (const expectedPath of expectedPaths) {
    const expected = await readRgba(expectedPath);
    const { width, height } = expected;
    const sizeMismatch = original.width !== width || original.height !== height;
    const actual = sizeMismatch
      ? await readRgba(
          await sharp(actualPng)
            .resize(width, height, { fit: "fill", kernel: "nearest" })
            .png()
            .toBuffer(),
        )
      : original;
    const diff = Buffer.alloc(width * height * 4);
    const mismatched = pixelmatch(
      actual.data,
      expected.data,
      diff,
      width,
      height,
      { threshold },
    );
    const diffRatio = mismatched / (width * height);
    if (!best || diffRatio < best.diffRatio) {
      best = {
        diffRatio,
        expectedPath,
        sizeMismatch,
        diff: { data: diff, width, height },
      };
    }
  }
  if (!best) throw new Error("No expected image");
  return best;
}

export function toPng(image: RgbaImage): Promise<Buffer> {
  return sharp(image.data, {
    raw: { width: image.width, height: image.height, channels: 4 },
  })
    .png()
    .toBuffer();
}
