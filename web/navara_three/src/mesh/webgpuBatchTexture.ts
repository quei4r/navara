import type { Material } from "three";

import {
  getBatchTextureLayout,
  getBatchTextureUniform,
  updateBatchAttribute,
  type BatchTextureSupport,
  type BatchVec3Key,
} from "../batchTexture";
import type { BatchScalarSlotKey } from "../batchTexture/types";

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * WebGPU-side helpers for the batch data texture (CPU-side layout and write
 * path live in batchTexture/). Two differences from the classic GLSL path
 * drive the design:
 *
 * - The GLSL chunks read layout defines stamped at write time; a TSL node
 *   graph is built once, so every supported row/slot must exist before the
 *   graph is built. eagerAllocateBatchTexture() forces allocation by writing
 *   each attribute's identity value once — the allocator backfills that
 *   identity to every batch and the texture reaches its final size, so the
 *   texture object captured by the graph is never swapped afterwards (growth
 *   only happens on new allocations).
 * - The classic path gates batch values behind USE_BATCH_* defines; with
 *   identity backfill, color (white), emissive (black × intensity 1) and
 *   packed show/opacity (visible, 1) are multiplicative/selection identities
 *   and are sampled unconditionally. height/extrudedHeight still need a gate
 *   uniform: their batch value REPLACES the material-level addHeight uniform
 *   (see height_vertex.glsl), and the zero default would clobber it. The
 *   mesh records first-write flags in its _updateBatchAttribute override.
 *   lineWidth/size carry a -1 fallback sentinel, so the graph selects the
 *   material default without any gate.
 */

/** Identity write per attribute, mirroring the allocator's backfill defaults. */
function identityValue(
  attribute: BatchVec3Key | BatchScalarSlotKey,
): number | number[] | boolean {
  switch (attribute) {
    case "color":
      return [1, 1, 1];
    case "emissive":
      return [0, 0, 0];
    case "emissiveIntensity":
      return 1;
    case "lineWidth":
    case "size":
      return -1;
    default:
      return 0;
  }
}

/**
 * Force-allocate every batch attribute the mesh type supports (plus the
 * packed show/opacity slot when `showOpacity` is set) so the TSL graph can
 * sample fixed rows/slots. No-op for writes the write path rejects.
 */
export function eagerAllocateBatchTexture(
  material: Material,
  support: BatchTextureSupport,
  opts: { showOpacity?: boolean } = {},
): void {
  for (const key of support.vec3s ?? ["color"]) {
    // An emissive write allocates the paired intensity slot too
    // (ensureEmissiveSlots), so no separate emissiveIntensity write is needed.
    updateBatchAttribute(material, 0, key, identityValue(key));
  }
  for (const key of support.scalars) {
    updateBatchAttribute(material, 0, key, identityValue(key));
  }
  if (opts.showOpacity) {
    updateBatchAttribute(material, 0, "show", true);
  }
}

export type WebgpuBatchSampler = {
  /** Texture node backing every fetch; re-point `.value` from the shared
   *  batch uniform ref if the texture is ever recreated. */
  texNode: { value: unknown };
  /** vec3 attribute row (color/emissive); null when not allocated. */
  vec3: (key: BatchVec3Key) => any;
  /** Scalar slot component; null when not allocated. */
  scalar: (key: BatchScalarSlotKey) => any;
  /** Packed show/opacity decoded to float nodes (see packShowOpacity). */
  showOpacity: () => { show: any; opacity: any } | null;
};

const COMPS = ["x", "y", "z", "w"] as const;

/**
 * Build TSL fetchers over the material's batch data texture, or null when
 * batching is unavailable (no texture — call eagerAllocateBatchTexture first —
 * or the geometry has no `_batchid` attribute, checked by the caller).
 *
 * Planar addressing (batchBaseIndex in batchTexture/layout.ts):
 *   col         = batchId % width
 *   physicalRow = attributeRow * groups + floor(batchId / width)
 * with width = min(batchLength, 4096) and groups = ceil(batchLength / width),
 * both fixed once batchLength is known — baked as graph constants here.
 * texel fetch (texture().load()) needs no sampler, works in the vertex stage,
 * and sidesteps float32 filtering being an optional WebGPU feature.
 */
export function createWebgpuBatchSampler(
  T: any,
  material: Material,
  batchLength: number,
  opts: { bid?: any } = {},
): WebgpuBatchSampler | null {
  const uniform = getBatchTextureUniform(material);
  const layout = getBatchTextureLayout(material);
  const texture = uniform?.value;
  if (!uniform || !layout || !texture || layout.rows === 0) return null;

  const width = texture.image.width;
  const groups = Math.ceil(batchLength / width);
  const texNode = T.texture(texture);
  // Default source is the `_batchid` attribute; a mesh without one (sdfText,
  // whose glyph instances get the feature index via the label data texture)
  // passes its own expression — the NVR_BATCH_ID_EXPR override idiom in
  // batch_texture_vertex.glsl.
  const bid: any = opts.bid ?? T.attribute("_batchid");
  const col = bid.mod(width).toUint();
  const batchRow = bid.div(width).floor().toUint();
  const fetchTexel = (attributeRow: number): any =>
    texNode.load(T.uvec2(col, batchRow.add(attributeRow * groups)));

  return {
    texNode: texNode as { value: unknown },
    vec3: (key) => {
      const row = layout.getVec3Row(key);
      return row == null ? null : fetchTexel(row).xyz;
    },
    scalar: (key) => {
      const slot = layout.getScalarSlot(key);
      return slot == null ? null : fetchTexel(slot.row)[COMPS[slot.comp]];
    },
    showOpacity: () => {
      const slot = layout.getScalarSlot("showOpacity");
      if (slot == null) return null;
      const packed = fetchTexel(slot.row)[COMPS[slot.comp]];
      return {
        show: T.step(0, packed),
        opacity: T.abs(packed).sub(1).clamp(0, 1),
      };
    },
  };
}

/**
 * Re-point the sampler's texture node if the shared batch uniform ref swapped
 * textures (defensive: eager allocation makes growth impossible, but a
 * disposed/recreated texture would otherwise leave the graph sampling dead
 * GPU memory). Call from the mesh's per-frame sync.
 */
export function syncWebgpuBatchTexture(
  sampler: { texNode: { value: unknown } },
  material: Material,
): void {
  const texture = getBatchTextureUniform(material)?.value;
  if (texture && sampler.texNode.value !== texture) {
    sampler.texNode.value = texture;
  }
}
