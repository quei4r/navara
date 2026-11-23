#![doc = include_str!("../README.md")]

use bevy_ecs::resource::Resource;

/// Cap on delegated worker tasks in flight across every task kind. The
/// frontend sets it from its pool size (`Core.setMaxWorkerTasks`) so the
/// pool's queue stays shallow and the kind order of `navara_worker`'s
/// dispatch chain (terrain mesh first) decides which task reaches a worker
/// next.
///
/// `external_load` (`Core.setWorkerExternalLoad`) is the pool work the
/// frontend runs outside delegated tasks (image decodes, Draco, descriptor
/// and plugin tasks, tasks cancelled mid-run); it takes from the same cap.
///
/// `intake_paused` (`Core.setWorkerIntakePaused`) stops everything that
/// would put work on the pool, e.g. while every pool worker is draining its
/// heap: `navara_worker` delegates nothing, `navara_feature` holds back model
/// adds and `navara_data_requester` holds back image fetches.
#[derive(Resource, Debug, Clone, Copy)]
pub struct WorkerLimits {
    pub max_pendings: u32,
    pub external_load: u32,
    pub intake_paused: bool,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            max_pendings: 16,
            external_load: 0,
            intake_paused: false,
        }
    }
}

impl WorkerLimits {
    /// The frontend paused worker intake: nothing that would put work on the
    /// worker pool should be handed to it.
    pub fn is_paused(&self) -> bool {
        self.intake_paused
    }
}
