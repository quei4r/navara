use bevy_ecs::system::{Commands, ResMut};
use navara_component::Priority;

use super::{ConstructPolygonBatchedFeatureMarker, ConstructPolygonBatchedFeatureParameters};
use crate::{
    DelegatedWorkerTask, DelegatedWorkerTasksParameters, DelegatedWorkerTasksParametersBundle,
    WorkerBudget,
    system::{WaitingTasks, delegate_nearest},
};

pub(crate) fn construct_polygon_batched_feature(
    commands: Commands,
    waiting: WaitingTasks<ConstructPolygonBatchedFeatureParameters>,
    budget: ResMut<WorkerBudget>,
) {
    delegate_nearest(commands, waiting, budget, |e, params| {
        DelegatedWorkerTasksParametersBundle::new(
            ConstructPolygonBatchedFeatureMarker,
            DelegatedWorkerTasksParameters::ConstructPolygonBatchedFeature(
                DelegatedWorkerTask::new(e, params.clone()),
            ),
            Priority::Medium,
        )
    });
}
