use bevy_ecs::system::{Commands, ResMut};
use navara_component::Priority;

use super::{ConstructPolylineBatchedFeatureMarker, ConstructPolylineBatchedFeatureParameters};
use crate::{
    DelegatedWorkerTask, DelegatedWorkerTasksParameters, DelegatedWorkerTasksParametersBundle,
    WorkerBudget,
    system::{WaitingTasks, delegate_nearest},
};

pub(crate) fn construct_polyline_batched_feature(
    commands: Commands,
    waiting: WaitingTasks<ConstructPolylineBatchedFeatureParameters>,
    budget: ResMut<WorkerBudget>,
) {
    delegate_nearest(commands, waiting, budget, |e, params| {
        DelegatedWorkerTasksParametersBundle::new(
            ConstructPolylineBatchedFeatureMarker,
            DelegatedWorkerTasksParameters::ConstructPolylineBatchedFeature(
                DelegatedWorkerTask::new(e, params.clone()),
            ),
            Priority::Medium,
        )
    });
}
