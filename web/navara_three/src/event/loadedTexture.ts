import type { Texture } from "three";

import { disposeTexture } from "../loaders";

import type { EventContext } from "./context";

/**
 * Releases everything a texture entity holds on removal: its hillshade
 * render target, temp DEM and pending edges, and the texture indexed in
 * `loadedTexs`. A hillshade entity's texture is the render target's own, so
 * `clearRenderTarget` has already disposed it. `fallback` is a texture that
 * never reached the index (a load that settled after its removal).
 */
export function releaseLoadedTexture(
  ctx: EventContext,
  id: string | undefined,
  fallback?: Texture,
) {
  const { loadedTexs, hillshadeContext } = ctx;
  if (hillshadeContext && id !== undefined) {
    hillshadeContext.pendingEdges.delete(id);
    hillshadeContext.clearTempDem(id);
    hillshadeContext.clearRenderTarget(id);
  }
  const tex = (id !== undefined ? loadedTexs.get(id) : undefined) ?? fallback;
  if (tex && !tex.isRenderTargetTexture) {
    disposeTexture(tex);
  }
  if (id !== undefined) loadedTexs.delete(id);
}
