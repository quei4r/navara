import { generate_id_from_entity, type JobHandler } from "@navaramap/core";
import {
  type EntityEvent,
  type TextureFragmentRequestedEvent,
  TextureFragmentStatus,
} from "@navaramap/engine";
import type { Texture } from "three";

import { disposeTexture } from "../loaders";

import type { EventContext } from "./context";
import { releaseLoadedTexture } from "./loadedTexture";
import { ABORTABLE_TEXTURE_LOADER } from "./loaders";

export const textureFragmentJobHandler = (
  ctx: EventContext,
): JobHandler<TextureFragmentRequestedEvent, never, EntityEvent, Texture> => ({
  name: "texture fragment",
  events: {
    requested: "texture_fragment_requested",
    removed: "texture_fragment_removed",
  },
  start: (ev, abort) => processTextureFragmentRequested(ctx, ev, abort),
  teardown: (texture, ev) => processTextureFragmentRemoved(ctx, texture, ev),
  failed: (ev) =>
    ctx.texFragment.triggerTextureFragmentLoaded(
      ev.bits,
      TextureFragmentStatus.Fail,
    ),
});

async function processTextureFragmentRequested(
  ctx: EventContext,
  req: TextureFragmentRequestedEvent,
  abort: AbortController,
): Promise<Texture | undefined> {
  const { texFragment, loadedTexs } = ctx;
  const id = generate_id_from_entity(req);
  if (loadedTexs.has(id)) return;

  const texture = await ABORTABLE_TEXTURE_LOADER.loadAsyncWithAbort(
    req.url,
    abort,
  );
  if (abort.signal.aborted) {
    disposeTexture(texture);
    return;
  }
  loadedTexs.set(id, texture);
  texFragment.triggerTextureFragmentLoaded(
    req.bits,
    TextureFragmentStatus.Success,
  );
  return texture;
}

function processTextureFragmentRemoved(
  ctx: EventContext,
  texture: Texture | undefined,
  req: EntityEvent | undefined,
) {
  releaseLoadedTexture(
    ctx,
    req ? generate_id_from_entity(req) : undefined,
    texture,
  );
}
