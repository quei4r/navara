import {
  Loader,
  Texture,
  type Material,
  type Mesh,
  type Object3D,
} from "three";

import { AbortableImageLoader } from "./AbortableImageLoader";

/**
 * Dispose a texture and release its backing image. When the texture is backed
 * by an `ImageBitmap` (the `createImageBitmap` decode path in
 * AbortableImageLoader), `texture.dispose()` frees the GL texture but NOT the
 * bitmap's decoded-pixel memory, which lives off-heap until `.close()` — so a
 * plain dispose leaks it on every tile removal. Safe for any texture: images
 * that are not ImageBitmaps (render targets, HTMLImageElement) are left alone.
 */
export function disposeTexture(texture: Texture): void {
  const image = texture.image as unknown;
  texture.dispose();
  if (typeof ImageBitmap !== "undefined" && image instanceof ImageBitmap) {
    image.close();
  }
}

/**
 * Deferred texture retirement for tile textures torn down by Rust removal
 * events. A removal can race tiles that still bind the texture (LOD churn:
 * ancestor-fallback slots, tile updates still in flight). Destroying the GPU
 * texture then is fatal on the WebGPU backend — direct texture bindings sample
 * the recreated empty texture (white tiles) — while WebGL masks it by keeping
 * the previously uploaded texels.
 *
 * Retirement closes the ImageBitmap immediately (its decoded pixels are the
 * bulk of the memory, and the already-uploaded GPU texture keeps sampling
 * without it), but defers `texture.dispose()` until a sweep finds the texture
 * bound nowhere. A texture that stays bound (a stale tile binding) is kept —
 * disposing it would reintroduce the white-tile regression.
 */
const RETIRE_SWEEP_MS = 5_000;
const RETIRE_GRACE_MS = 10_000;

type TraversableScenes = Record<
  string,
  { traverse: (cb: (object: Object3D) => void) => void } | undefined
>;

const retired: { texture: Texture; at: number }[] = [];
let sweepTimer: ReturnType<typeof setInterval> | undefined;
let retiredScenes: (() => TraversableScenes) | undefined;

export function retireTexture(
  texture: Texture,
  scenes: () => TraversableScenes,
): void {
  const image = texture.image as unknown;
  if (
    typeof ImageBitmap !== "undefined" &&
    image instanceof ImageBitmap &&
    image.width > 0
  ) {
    image.close();
  }
  retired.push({ texture, at: Date.now() });
  retiredScenes = scenes;
  sweepTimer ??= setInterval(sweepRetiredTextures, RETIRE_SWEEP_MS);
}

function collectBoundTextures(scenes: TraversableScenes): Set<Texture> {
  const bound = new Set<Texture>();
  const addMaterial = (material: Material | Material[] | undefined) => {
    if (!material || Array.isArray(material)) return;
    const m = material as Material & { map?: Texture | null };
    if (m.map) bound.add(m.map);
    const ud = m.userData as {
      textures?: { value?: (Texture | null)[] };
      webgpuSlots?: { node?: { value?: Texture } }[];
      webgpuHillshade?: { node?: { value?: Texture } };
    };
    for (const t of ud.textures?.value ?? []) {
      if (t) bound.add(t);
    }
    for (const slot of ud.webgpuSlots ?? []) {
      const v = slot?.node?.value;
      if (v) bound.add(v);
    }
    const hs = ud.webgpuHillshade?.node?.value;
    if (hs) bound.add(hs);
  };
  for (const scene of Object.values(scenes)) {
    scene?.traverse((o) => addMaterial((o as Mesh).material));
  }
  return bound;
}

function sweepRetiredTextures(): void {
  if (!retired.length) {
    clearInterval(sweepTimer);
    sweepTimer = undefined;
    return;
  }
  const now = Date.now();
  if (!retired.some((e) => now - e.at >= RETIRE_GRACE_MS)) return;
  const bound = collectBoundTextures(retiredScenes?.() ?? {});
  for (let i = retired.length - 1; i >= 0; i--) {
    const entry = retired[i];
    if (now - entry.at < RETIRE_GRACE_MS) continue;
    if (bound.has(entry.texture)) continue;
    entry.texture.dispose();
    retired.splice(i, 1);
  }
}

/** Force-dispose every retired texture (view teardown). */
export function flushRetiredTextures(): void {
  clearInterval(sweepTimer);
  sweepTimer = undefined;
  for (const entry of retired.splice(0)) {
    entry.texture.dispose();
  }
}

// Ref: https://github.com/mrdoob/three.js/blob/beab9e845f9e5ae11d648f55b24a0e910b56a85a/src/loaders/TextureLoader.js
export class AbortableTextureLoader extends Loader {
  loadAsyncWithAbort(
    url: string,
    abort?: AbortController,
    onProgress?: (event: ProgressEvent) => void,
  ) {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const scope = this;

    return new Promise<Texture>(function (resolve, reject) {
      scope.load(url, resolve, onProgress, reject, abort);
    });
  }

  load(
    url: string,
    onLoad: (data: Texture) => void,
    onProgress?: (event: ProgressEvent) => void,
    onError?: (err: unknown, isAborted?: boolean) => void,
    abort?: AbortController,
  ): Texture {
    const texture = new Texture();

    const loader = new AbortableImageLoader(this.manager);
    loader.setCrossOrigin(this.crossOrigin);
    loader.setPath(this.path);

    loader.load(
      url,
      function (image) {
        texture.image = image;
        if (
          typeof ImageBitmap !== "undefined" &&
          image instanceof ImageBitmap
        ) {
          // The bitmap is already flipped at decode time
          // (imageOrientation: "flipY" — see AbortableImageLoader).
          texture.flipY = false;
        }
        texture.needsUpdate = true;

        if (onLoad !== undefined) {
          onLoad(texture);
        }
      },
      onProgress,
      onError,
      abort,
    );

    return texture;
  }
}
