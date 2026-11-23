import { generate_id_from_entity, type JobHandler } from "@navaramap/core";
import {
  type EntityEvent,
  type MeshAdded,
  MeshChanged,
  type MeshGeometryReplaced,
  type TerrainExaggerationUpdatedEvent,
} from "@navaramap/engine";

import { TileMesh } from "../mesh";

import type { EventContext } from "./context";
import { processObjectRemoved } from "./objectRemoved";

export const tileMeshJobHandler = (
  ctx: EventContext,
): JobHandler<MeshAdded, MeshChanged, EntityEvent, TileMesh> => ({
  name: "tile mesh",
  events: {
    requested: "mesh_added",
    changed: "mesh_updated",
    removed: "mesh_removed",
  },
  start: (ev, abort) => processMeshAdded(ctx, ev, abort),
  change: (m, ev) => {
    // A mesh whose geometry could not be installed never entered the scene
    // or the index; it is kept only so teardown releases what it acquired.
    if (ctx.meshes.get(generate_id_from_entity(ev)) === m) m._update(ev);
  },
  teardown: (m, ev) => processObjectRemoved(ctx, m, ev),
  // The engine gates the tile's LOD swap on the prepared report and nothing
  // else re-triggers it, so a mesh that could not be built still reports:
  // the swap goes ahead without the mesh instead of stalling until eviction.
  failed: (ev, error) => {
    console.error("Tile mesh failed to build:", error);
    ctx.meshHandler.setTileMeshPrepared(ev.tile_handle);
  },
});

export async function processMeshAdded(
  ctx: EventContext,
  mesh: MeshAdded,
  abort: AbortController,
): Promise<TileMesh> {
  const m = new TileMesh(ctx, mesh);
  try {
    await m._init(mesh);
  } catch (error) {
    // A rejected start leaves teardown without a product; release it here.
    processObjectRemoved(ctx, m, mesh);
    throw error;
  }
  // The engine gates the tile's LOD swap on this report; a removed tile has
  // no swap to wait for.
  if (!abort.signal.aborted) {
    ctx.meshHandler.setTileMeshPrepared(mesh.tile_handle);
  }
  return m;
}

export function processMeshGeometryReplaced(
  ctx: EventContext,
  ev: MeshGeometryReplaced,
) {
  const id = generate_id_from_entity(ev);
  const m = ctx.meshes.get(id);
  if (!m || !(m instanceof TileMesh)) return;

  m.replaceGeometry(ev.mesh);
}

/**
 * The GPU displaces terrain vertices through the shared exaggeration uniform
 * and horizon-culls against the shared shrunk ellipsoid, so a change only
 * needs those uniforms and the tiles' culling bounds updated.
 */
export function processTerrainExaggerationUpdated(
  ctx: EventContext,
  ev: TerrainExaggerationUpdatedEvent | undefined,
) {
  if (!ev) return;
  ctx.uniforms.terrainExaggeration.value = [ev.scale, ev.relative_height];
  ctx.uniforms.horizonMinHeight.value = ev.horizon_minimum_height;
  for (const tile of ctx.tileMapByHandle.values()) {
    tile.updateTerrainExaggeration();
  }
}

/** Point the terrain tiles Rust handed new vector drape sources at them. */
export function processVectorDrapeChanged(
  ctx: EventContext,
  handles: BigUint64Array,
) {
  for (const handle of handles) {
    ctx.tileMapByHandle.get(handle)?.markVectorDrapeSourcesChanged();
  }
}
