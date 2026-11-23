mod component;
mod delegated_task;
mod event;
mod system;
mod tasks;

use std::any::{TypeId, type_name};

use bevy_app::Update;
use bevy_ecs::schedule::IntoScheduleConfigs;
pub use component::*;
pub use delegated_task::*;
pub use event::*;
pub use navara_worker_component::WorkerLimits;
use rustc_hash::FxHashSet;
pub use system::WorkerBudget;
pub use tasks::*;

pub struct WorkerPlugin;

/// One line per delegated task kind, most urgent first: `marker => system`.
/// Expands to every kind's slot reservation, then every kind's delegation,
/// both in this order, so the two passes can never disagree on priority.
/// Panics while the plugin builds if a marker is listed twice.
macro_rules! delegate_by_priority {
    ($($marker:ty => $delegate:path),* $(,)?) => {{
        assert_distinct_kinds(&[$((TypeId::of::<$marker>(), type_name::<$marker>()),)*]);
        (
            ($(system::reserve_for_starving_kind::<$marker>,)*).chain(),
            ($($delegate,)*).chain(),
        )
            .chain()
    }};
}

/// A kind listed twice would delegate twice per frame and take its reserved
/// slot twice.
fn assert_distinct_kinds(kinds: &[(TypeId, &str)]) {
    let mut seen = FxHashSet::default();
    for (id, name) in kinds {
        assert!(
            seen.insert(*id),
            "worker task kind `{name}` is listed twice in delegate_by_priority!"
        );
    }
}

impl bevy_app::Plugin for WorkerPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        #[cfg(not(feature = "delegated_worker"))]
        {
            use bevy_app::PreUpdate;
            app.add_message::<WorkerTaskCompletedEvent>().add_systems(
                PreUpdate,
                tasks::construct_terrain_mesh::system::setup_martini,
            );
        }

        app.add_message::<WorkerTaskCompletedEvent>()
            .add_message::<WorkerTaskFailedEvent>()
            .init_resource::<WorkerLimits>()
            .init_resource::<WorkerBudget>()
            .add_systems(
            Update,
            (
                system::handle_failed_event,
                system::remove,
                // Before the budget, so a completed task's delegatee (marked
                // `Deleted` here) frees its slot this frame.
                system::handle_completed_event,
                system::compute_worker_budget,
                delegate_by_priority![
                    construct_terrain_mesh::ConstructTerrainMeshMarker
                        => tasks::construct_terrain_mesh::system::construct_terrain_mesh,
                    upsample_terrain_mesh::UpsampleTerrainMeshMarker
                        => tasks::upsample_terrain_mesh::system::upsample_terrain_mesh,
                    parse_mvt_tile::ParseMvtTileMarker
                        => tasks::parse_mvt_tile::system::parse_mvt_tile,
                    construct_polygon_batched_feature::ConstructPolygonBatchedFeatureMarker
                        => tasks::construct_polygon_batched_feature::system::construct_polygon_batched_feature,
                    construct_polyline_batched_feature::ConstructPolylineBatchedFeatureMarker
                        => tasks::construct_polyline_batched_feature::system::construct_polyline_batched_feature,
                ],
                system::commit,
            )
                .chain(),
        );
    }
}

#[cfg(test)]
mod test {
    use std::any::{TypeId, type_name};

    use super::*;

    #[test]
    fn it_should_build_with_every_kind_listed_once() {
        bevy_app::App::new().add_plugins(WorkerPlugin);
    }

    #[test]
    #[should_panic(expected = "listed twice")]
    fn it_should_reject_a_kind_listed_twice() {
        type Mvt = parse_mvt_tile::ParseMvtTileMarker;
        assert_distinct_kinds(&[
            (TypeId::of::<Mvt>(), type_name::<Mvt>()),
            (TypeId::of::<Mvt>(), type_name::<Mvt>()),
        ]);
    }
}
