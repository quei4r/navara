//! Hands the resolved vector drape sources of the shown terrain tiles over to the frontend,
//! which re-bakes a terrain tile's drape whenever its sources change. A vector tile that
//! becomes a drape source switches every terrain tile it covers in the same frame, so the
//! switches are spread over frames instead of all reaching the frontend at once.

use std::collections::VecDeque;

use bevy_ecs::{prelude::*, system::SystemParam};
use navara_component::Deleted;
use navara_event_store::EventStore;
use navara_feature_component::{id::FeatureId, render::RenderableFeature};
use navara_layer::LayerDescStore;
use navara_mesh::Mesh;
use navara_tile_component::{TerrainTileQuadtree, TileHandle, TileMeshMarker, VectorTileQuadtree};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    LayerResources, RenderedTile, ResolvedVectorTileState, TileCacheManager, VectorResolveRevision,
    VectorTileFeatureMarker, VectorTileSourceResources, resolve_vector_tile_states,
};

/// Drape source switches handed to the frontend per frame. Each re-bakes one terrain tile's
/// drape.
pub const DRAPE_SOURCE_SWITCHES_PER_FRAME: usize = 8;

/// The vector drape sources handed to the frontend, per terrain tile.
#[derive(Resource, Default)]
pub struct VectorDrapeSources {
    handed: FxHashMap<TileHandle, Vec<ResolvedVectorTileState>>,
    /// Shown terrain tiles whose resolved sources differ from the handed ones, oldest first.
    waiting: VecDeque<TileHandle>,
}

impl VectorDrapeSources {
    /// The sources handed over for `terrain`.
    pub fn handed(&self, terrain: TileHandle) -> &[ResolvedVectorTileState] {
        self.handed.get(&terrain).map_or(&[], Vec::as_slice)
    }

    /// Hand the shown terrain tiles' `resolved` sources over, appending the tiles whose handed
    /// sources changed to `changed`. A tile with nothing handed yet gets its sources at once,
    /// and so does a tile whose handed sources are not all `held` any more, since the frontend
    /// would re-bake it from a missing scene. Other switches wait in FIFO order, `budget` per
    /// call, while the frontend keeps the last bake. A waiting tile that is not shown, or whose
    /// sources switched back, leaves the queue.
    pub fn hand_over(
        &mut self,
        resolved: Vec<(TileHandle, Vec<ResolvedVectorTileState>)>,
        held: impl Fn(&ResolvedVectorTileState) -> bool,
        budget: usize,
        changed: &mut Vec<TileHandle>,
    ) {
        let mut switching = FxHashMap::default();
        let mut arrived = Vec::new();
        for (terrain, sources) in resolved {
            match self.handed.get(&terrain) {
                Some(handed) if *handed == sources => {}
                Some(handed) if !handed.is_empty() && handed.iter().all(&held) => {
                    arrived.push(terrain);
                    switching.insert(terrain, sources);
                }
                _ => {
                    self.handed.insert(terrain, sources);
                    changed.push(terrain);
                }
            }
        }

        self.waiting.retain(|t| switching.contains_key(t));
        let mut queued = FxHashSet::default();
        for t in &self.waiting {
            queued.insert(*t);
        }
        for t in arrived {
            if !queued.contains(&t) {
                self.waiting.push_back(t);
            }
        }

        for _ in 0..budget {
            let Some(terrain) = self.waiting.pop_front() else {
                break;
            };
            if let Some(sources) = switching.remove(&terrain) {
                self.handed.insert(terrain, sources);
                changed.push(terrain);
            }
        }
    }

    /// Forget `terrain`, so its next sources are handed at once.
    pub fn forget(&mut self, terrain: TileHandle) {
        self.handed.remove(&terrain);
        self.waiting.retain(|t| *t != terrain);
    }
}

/// Resolve the drape sources of the shown terrain tiles from the draped (clamp-to-ground)
/// layers, keep the ones with a draped feature, and hand them over (see
/// [`VectorDrapeSources::hand_over`]), listing the terrain tiles whose sources changed in
/// the frame's events.
#[allow(clippy::too_many_arguments)]
pub fn hand_over_vector_drape_sources(
    mut drape_sources: ResMut<VectorDrapeSources>,
    mut events: ResMut<EventStore>,
    revision: Res<VectorResolveRevision>,
    layer_order: Option<Res<LayerDescStore>>,
    terrain: Res<TerrainTileQuadtree>,
    layers: Query<&LayerResources>,
    sources: Query<&VectorTileSourceResources>,
    qts: Query<&VectorTileQuadtree>,
    tcs: Query<&TileCacheManager>,
    tile_features: RenderedTileFeatures,
    mut destroyed_tiles: RemovedComponents<RenderedTile>,
    meshes: Query<(&TileMeshMarker, &Mesh), Without<Deleted>>,
    added_meshes: Query<&TileMeshMarker, Added<TileMeshMarker>>,
    changed_meshes: Query<(), (With<TileMeshMarker>, Changed<Mesh>)>,
) {
    let layers_reordered = layer_order.as_ref().is_some_and(|s| s.is_changed());
    // A destroyed (e.g. evicted) source tile leaves its quadtree without a traverse
    // recording a new drape source.
    let tiles_destroyed = !destroyed_tiles.is_empty();
    destroyed_tiles.clear();
    if !revision.is_changed()
        && !layers_reordered
        && !tiles_destroyed
        && drape_sources.waiting.is_empty()
        && added_meshes.is_empty()
        && changed_meshes.is_empty()
    {
        return;
    }
    // A new mesh starts without a bake, even when its tile had one before.
    for marker in &added_meshes {
        drape_sources.forget(marker.handle);
    }

    // Draped layers in declaration order, so the frontend stacks them like raster layers.
    let mut draped = Vec::new();
    for layer in &layers {
        let Ok(source) = sources.get(layer.source) else {
            continue;
        };
        if !source.source_id.traversal_config().has_clamp_to_ground {
            continue;
        }
        let (Ok(qt), Ok(tc)) = (qts.get(layer.quadtree), tcs.get(layer.tile_cache_manager)) else {
            continue;
        };
        let order = layer_order
            .as_ref()
            .and_then(|s| s.get_order(&layer.layer_id).copied())
            .unwrap_or(usize::MAX);
        draped.push((order, layer.layer_id.clone(), qt, tc));
    }
    draped.sort_by_key(|(order, ..)| *order);
    let mut quadtrees = Vec::with_capacity(draped.len());
    let mut caches = FxHashMap::default();
    for (_, layer_id, qt, tc) in &draped {
        quadtrees.push((layer_id.clone(), *qt));
        caches.insert(layer_id.as_str(), *tc);
    }

    // The frontend bakes only draped features, so a source without one takes no drape slot.
    let mut drapes = FxHashMap::default();
    let mut alive = FxHashSet::default();
    let mut resolved = Vec::new();
    for (marker, mesh) in &meshes {
        alive.insert(marker.handle);
        if !mesh.active {
            continue;
        }
        let Some(tile) = terrain.qt.get(marker.handle) else {
            continue;
        };
        let mut states =
            resolve_vector_tile_states(tile.extent, tile.tiling_scheme.is_geographic(), &quadtrees);
        states.retain(|s| {
            *drapes
                .entry((s.layer_id.clone(), s.tile_handle))
                .or_insert_with(|| {
                    caches
                        .get(s.layer_id.as_str())
                        .is_some_and(|tc| tile_features.has_draped_feature(tc, s.tile_handle))
                })
        });
        resolved.push((marker.handle, states));
    }
    drape_sources.handed.retain(|t, _| alive.contains(t));

    // The frontend holds a source's scene until the source tile is evicted.
    let held = |s: &ResolvedVectorTileState| {
        caches
            .get(s.layer_id.as_str())
            .is_some_and(|tc| tc.rendered_tile_caches.contains_key(&s.tile_handle))
    };
    drape_sources.hand_over(
        resolved,
        held,
        DRAPE_SOURCE_SWITCHES_PER_FRAME,
        &mut events.vector_drape_changed,
    );
}

/// The features of the rendered vector tiles.
#[derive(SystemParam)]
pub struct RenderedTileFeatures<'w, 's> {
    rendered_tiles: Query<'w, 's, &'static RenderedTile>,
    features: Query<'w, 's, &'static FeatureId, With<VectorTileFeatureMarker>>,
    renderable_features: Query<'w, 's, &'static RenderableFeature>,
}

impl RenderedTileFeatures<'_, '_> {
    /// Whether the rendered vector tile `handle` has a draped feature.
    fn has_draped_feature(&self, tc: &TileCacheManager, handle: TileHandle) -> bool {
        let Some(feature_ids) = tc
            .rendered_tile_caches
            .get(&handle)
            .and_then(|e| self.rendered_tiles.get(*e).ok())
            .and_then(|t| t.feature_ids.as_ref())
        else {
            return false;
        };
        for id in feature_ids {
            if let Ok(FeatureId(Some(feature))) = self.features.get(*id)
                && self
                    .renderable_features
                    .get(*feature)
                    .is_ok_and(RenderableFeature::is_draped)
            {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::system::SystemState;
    use navara_feature_component::render::PolygonRenderInformation;

    use super::*;

    fn source(tile_handle: TileHandle) -> ResolvedVectorTileState {
        ResolvedVectorTileState {
            layer_id: "roads".to_string(),
            tile_handle,
            uv_offset: [0., 0.],
            uv_scale: [1., 1.],
            reproject_terrain_lat: None,
        }
    }

    fn hand_over(
        sources: &mut VectorDrapeSources,
        resolved: &[(TileHandle, TileHandle)],
        budget: usize,
    ) -> Vec<TileHandle> {
        let resolved = resolved
            .iter()
            .map(|(terrain, src)| (*terrain, vec![source(*src)]))
            .collect();
        let mut changed = Vec::new();
        sources.hand_over(resolved, |_| true, budget, &mut changed);
        changed
    }

    #[test]
    fn it_should_hand_first_sources_over_at_once() {
        let mut sources = VectorDrapeSources::default();

        assert_eq!(
            hand_over(&mut sources, &[(1, 10), (2, 10), (3, 10)], 1),
            [1, 2, 3]
        );
        assert_eq!(sources.handed(1)[0].tile_handle, 10);
    }

    #[test]
    fn it_should_hand_switches_over_within_the_budget_oldest_first() {
        let mut sources = VectorDrapeSources::default();
        hand_over(&mut sources, &[(1, 10), (2, 10), (3, 10)], 1);

        assert_eq!(
            hand_over(&mut sources, &[(1, 11), (2, 11), (3, 11)], 2),
            [1, 2]
        );
        assert_eq!(sources.handed(3)[0].tile_handle, 10);
        // Tile 3 waited longer than tile 1's new switch.
        assert_eq!(
            hand_over(&mut sources, &[(1, 12), (2, 11), (3, 11)], 1),
            [3]
        );
        assert_eq!(
            hand_over(&mut sources, &[(1, 12), (2, 11), (3, 11)], 1),
            [1]
        );
        assert!(sources.waiting.is_empty());
    }

    #[test]
    fn it_should_drop_a_waiting_tile_that_is_hidden_or_switched_back() {
        let mut sources = VectorDrapeSources::default();
        hand_over(&mut sources, &[(1, 10), (2, 10)], 1);
        assert!(hand_over(&mut sources, &[(1, 11), (2, 11)], 0).is_empty());

        // Tile 1 is hidden and tile 2 resolves to its handed source again.
        assert!(hand_over(&mut sources, &[(2, 10)], 1).is_empty());
        assert!(sources.waiting.is_empty());
        assert_eq!(sources.handed(1)[0].tile_handle, 10);
    }

    #[test]
    fn it_should_hand_a_switch_away_from_a_released_source_over_at_once() {
        let mut sources = VectorDrapeSources::default();
        hand_over(&mut sources, &[(1, 10), (2, 20)], 1);

        let mut changed = Vec::new();
        sources.hand_over(
            vec![(1, vec![source(11)]), (2, vec![source(21)])],
            |s| s.tile_handle != 20,
            0,
            &mut changed,
        );

        assert_eq!(changed, [2]);
        assert_eq!(sources.handed(2)[0].tile_handle, 21);
    }

    #[test]
    fn it_should_hand_sources_over_at_once_after_the_tile_is_forgotten() {
        let mut sources = VectorDrapeSources::default();
        hand_over(&mut sources, &[(1, 10)], 1);
        assert!(hand_over(&mut sources, &[(1, 11)], 0).is_empty());

        sources.forget(1);

        assert_eq!(hand_over(&mut sources, &[(1, 11)], 0), [1]);
        assert!(sources.waiting.is_empty());
    }

    fn polygon(draped: bool) -> RenderableFeature {
        RenderableFeature::Polygon {
            coordinates: Default::default(),
            crs: Default::default(),
            active: true,
            material: Default::default(),
            geometry: Default::default(),
            outline_geometry: None,
            transform: Default::default(),
            feature_id: Entity::PLACEHOLDER,
            render_info: PolygonRenderInformation {
                should_be_texturized: draped,
                ..Default::default()
            },
            extent: None,
            bounding_sphere: None,
            feature_batch_id: 0,
            batch_length: 0,
        }
    }

    fn spawn_rendered_tile(
        world: &mut World,
        tc: &mut TileCacheManager,
        handle: TileHandle,
        renderables: Vec<RenderableFeature>,
    ) {
        let mut feature_ids = Vec::new();
        for renderable in renderables {
            let renderable = world.spawn(renderable).id();
            feature_ids.push(
                world
                    .spawn((FeatureId(Some(renderable)), VectorTileFeatureMarker))
                    .id(),
            );
        }
        let tile = world
            .spawn(RenderedTile {
                tile_handle: handle,
                feature_ids: Some(feature_ids),
            })
            .id();
        tc.rendered_tile_caches.insert(handle, tile);
    }

    #[test]
    fn it_should_find_a_draped_feature_only_on_a_rendered_tile_with_one() {
        let mut world = World::new();
        let mut tc = TileCacheManager::default();
        spawn_rendered_tile(&mut world, &mut tc, 1, vec![polygon(false), polygon(true)]);
        spawn_rendered_tile(&mut world, &mut tc, 2, vec![polygon(false)]);
        spawn_rendered_tile(&mut world, &mut tc, 3, vec![]);

        let mut state: SystemState<RenderedTileFeatures> = SystemState::new(&mut world);
        let tile_features = state.get(&world).unwrap();
        let drapes = |handle| tile_features.has_draped_feature(&tc, handle);

        assert!(drapes(1));
        assert!(!drapes(2));
        assert!(!drapes(3));
        // Tile 4 is not rendered.
        assert!(!drapes(4));
    }
}
