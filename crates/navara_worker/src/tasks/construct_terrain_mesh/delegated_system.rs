use bevy_ecs::system::{Commands, ResMut};
use navara_component::Priority;

use super::{ConstructTerrainMeshMarker, ConstructTerrainMeshParameters};
use crate::{
    DelegatedWorkerTask, DelegatedWorkerTasksParameters, DelegatedWorkerTasksParametersBundle,
    WorkerBudget,
    system::{WaitingTasks, delegate_nearest},
};

pub(crate) fn construct_terrain_mesh(
    commands: Commands,
    waiting: WaitingTasks<ConstructTerrainMeshParameters>,
    budget: ResMut<WorkerBudget>,
) {
    delegate_nearest(commands, waiting, budget, |e, params| {
        DelegatedWorkerTasksParametersBundle::new(
            ConstructTerrainMeshMarker,
            DelegatedWorkerTasksParameters::ConstructTerrainMesh(DelegatedWorkerTask::new(
                e,
                params.clone(),
            )),
            Priority::High,
        )
    });
}
