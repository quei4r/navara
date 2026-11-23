use bevy_ecs::system::{Commands, ResMut};
use navara_component::Priority;

use super::{ParseMvtTileMarker, ParseMvtTileParameters};
use crate::{
    DelegatedWorkerTask, DelegatedWorkerTasksParameters, DelegatedWorkerTasksParametersBundle,
    WorkerBudget,
    system::{WaitingTasks, delegate_nearest},
};

pub(crate) fn parse_mvt_tile(
    commands: Commands,
    waiting: WaitingTasks<ParseMvtTileParameters>,
    budget: ResMut<WorkerBudget>,
) {
    delegate_nearest(commands, waiting, budget, |e, params| {
        DelegatedWorkerTasksParametersBundle::new(
            ParseMvtTileMarker,
            DelegatedWorkerTasksParameters::ParseMvtTile(DelegatedWorkerTask::new(
                e,
                params.clone(),
            )),
            Priority::Medium,
        )
    });
}

#[cfg(test)]
mod test {
    use bevy_ecs::system::RunSystemOnce;
    use bevy_ecs::world::World;
    use navara_component::{OrderByDistance, Requested};

    use super::*;
    use crate::WorkerTaskMarker;

    fn params() -> ParseMvtTileParameters {
        ParseMvtTileParameters {
            pbf_handle: 0,
            x: 0,
            y: 0,
            z: 0,
            tile_extent: None,
            configs: vec![],
            compression: 0,
        }
    }

    /// Delegation draws the shared budget down, nearest candidates first, and
    /// leaves the rest for the next frame.
    #[test]
    fn it_should_delegate_within_the_shared_budget_nearest_first() {
        let mut world = World::new();
        world.insert_resource(WorkerBudget {
            shared: 2,
            ..Default::default()
        });
        let far = world
            .spawn((
                WorkerTaskMarker,
                params(),
                OrderByDistance {
                    sse: 1.0,
                    distance: 30.0,
                },
            ))
            .id();
        let near = world
            .spawn((
                WorkerTaskMarker,
                params(),
                OrderByDistance {
                    sse: 1.0,
                    distance: 10.0,
                },
            ))
            .id();
        let mid = world
            .spawn((
                WorkerTaskMarker,
                params(),
                OrderByDistance {
                    sse: 1.0,
                    distance: 20.0,
                },
            ))
            .id();

        world.run_system_once(parse_mvt_tile).unwrap();

        assert!(world.get::<Requested>(near).is_some());
        assert!(world.get::<Requested>(mid).is_some());
        assert!(world.get::<Requested>(far).is_none());
        assert_eq!(world.resource::<WorkerBudget>().shared, 0);

        // Nothing is delegated once the budget is spent.
        world.run_system_once(parse_mvt_tile).unwrap();
        assert!(world.get::<Requested>(far).is_none());
    }
}
