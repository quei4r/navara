use bevy_ecs::system::{Commands, ResMut};
use navara_component::Priority;

use super::{UpsampleTerrainMeshMarker, UpsampleTerrainMeshParameters};
use crate::{
    DelegatedWorkerTask, DelegatedWorkerTasksParameters, DelegatedWorkerTasksParametersBundle,
    WorkerBudget,
    system::{WaitingTasks, delegate_nearest},
};

pub(crate) fn upsample_terrain_mesh(
    commands: Commands,
    waiting: WaitingTasks<UpsampleTerrainMeshParameters>,
    budget: ResMut<WorkerBudget>,
) {
    delegate_nearest(commands, waiting, budget, |e, params| {
        DelegatedWorkerTasksParametersBundle::new(
            UpsampleTerrainMeshMarker,
            DelegatedWorkerTasksParameters::UpsampleTerrainMesh(DelegatedWorkerTask::new(
                e,
                params.clone(),
            )),
            Priority::High,
        )
    });
}
