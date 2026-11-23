use std::any::TypeId;

use rustc_hash::FxHashSet;

use bevy_ecs::{
    component::Component,
    entity::Entity,
    message::MessageReader,
    query::{Added, Has, Or, With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use navara_buffer_store::BufferStore;
use navara_component::{Deleted, Priority, Requested};
use navara_event_store::EventStore;
use navara_worker_component::WorkerLimits;

use crate::{
    DelegatedWorkerTask, DelegatedWorkerTaskMarker, DelegatedWorkerTasksParameters,
    DelegatedWorkerTasksResult, WorkerTaskCompleted, WorkerTaskCompletedEvent,
    WorkerTaskDelegateeMarker, WorkerTaskFailedEvent,
    component::{FreeResultBuffers, WorkerTaskMarker},
};

/// Delegations still allowed this frame; recomputed by
/// [`compute_worker_budget`], partly reserved by [`reserve_for_starving_kind`]
/// and drawn down by each kind's delegating system in chain order.
///
/// A kind with waiting tasks and nothing in flight gets one slot reserved
/// before any kind takes from the shared rest, so a steady stream of the
/// kinds earlier in the chain cannot starve it, and the total in flight
/// never exceeds [`WorkerLimits::max_pendings`].
///
/// A kind is identified by its marker component, which both its delegator
/// and its delegatee carry (e.g. `ParseMvtTileMarker`).
#[derive(Resource, Debug, Default, Clone)]
pub struct WorkerBudget {
    /// Unreserved delegations left, open to every kind in chain order.
    pub shared: u32,
    /// Marker types of the kinds holding their reserved slot this frame.
    pub(crate) reserved: FxHashSet<TypeId>,
    /// The frontend paused intake (e.g. while every pool
    /// worker is draining its heap); nothing is delegated.
    pub paused: bool,
}

impl WorkerBudget {
    /// How many tasks of kind `M` may be delegated now.
    pub fn allowance<M: Component>(&self) -> usize {
        if self.paused {
            return 0;
        }
        self.shared as usize + usize::from(self.is_reserved::<M>())
    }

    /// Record `taken` delegations of kind `M`, its reserved slot first.
    pub fn spend<M: Component>(&mut self, taken: u32) {
        let mut taken = taken;
        if taken > 0 && self.release::<M>() {
            taken -= 1;
        }
        self.shared = self.shared.saturating_sub(taken);
    }

    /// Called at most once per kind and frame: `WorkerPlugin` rejects a kind
    /// listed twice when it builds.
    fn reserve<M: Component>(&mut self) {
        if self.paused || self.shared == 0 {
            return;
        }
        self.shared -= 1;
        self.reserved.insert(TypeId::of::<M>());
    }

    fn is_reserved<M: Component>(&self) -> bool {
        self.reserved.contains(&TypeId::of::<M>())
    }

    fn release<M: Component>(&mut self) -> bool {
        self.reserved.remove(&TypeId::of::<M>())
    }
}

/// Delegators of task kind `P` not yet handed to the platform.
#[cfg(feature = "delegated_worker")]
pub(crate) type WaitingTasks<'w, 's, P> = Query<
    'w,
    's,
    (
        Entity,
        &'static P,
        &'static navara_component::OrderByDistance,
    ),
    (With<WorkerTaskMarker>, Without<Requested>, Without<Deleted>),
>;

/// Delegate the nearest waiting tasks of kind `M` that the shared budget
/// still allows this frame; `spawn_delegatee` builds each task's delegatee.
#[cfg(feature = "delegated_worker")]
pub(crate) fn delegate_nearest<M: Component, P: Component>(
    mut commands: Commands,
    waiting: WaitingTasks<P>,
    mut budget: ResMut<WorkerBudget>,
    spawn_delegatee: impl Fn(Entity, &P) -> crate::DelegatedWorkerTasksParametersBundle<M>,
) {
    let num_take = budget.allowance::<M>();
    // An exhausted budget is the common case; skip sorting every waiting task.
    if num_take == 0 {
        return;
    }
    let mut taken = 0u32;
    for (e, params, _) in waiting
        .iter()
        .sort::<&navara_component::OrderByDistance>()
        .take(num_take)
    {
        taken += 1;
        let delegatee_id = commands.spawn(spawn_delegatee(e, params)).id();
        commands
            .entity(e)
            .insert((WorkerTaskDelegateeMarker(delegatee_id), Requested));
    }
    budget.spend::<M>(taken);
}

/// Reserve one slot of the shared budget for kind `M` when it has waiting
/// tasks and none in flight. Runs once per kind, in priority order, after
/// [`compute_worker_budget`] and before any kind delegates.
#[allow(clippy::type_complexity)]
pub fn reserve_for_starving_kind<M: Component>(
    mut budget: ResMut<WorkerBudget>,
    waiting: Query<
        (),
        (
            With<M>,
            With<WorkerTaskMarker>,
            Without<Requested>,
            Without<Deleted>,
        ),
    >,
    in_flight: Query<(), (With<M>, With<DelegatedWorkerTaskMarker>, Without<Deleted>)>,
) {
    if !waiting.is_empty() && in_flight.is_empty() {
        budget.reserve::<M>();
    }
}

/// Delegatees are spawned through deferred commands, so a kind's system
/// cannot see what an earlier kind delegated this frame: the shared budget
/// is the only count that stays exact across the chain.
///
/// The pool work the frontend runs outside delegated tasks
/// ([`WorkerLimits::external_load`]) takes from the same limit, so a burst
/// of image decodes cannot queue the next terrain mesh behind it.
///
/// A delegatee whose delegator no longer exists (despawned without passing
/// through [`remove`]) is marked `Deleted` here instead of being counted, so
/// it can never hold a slot of the shared budget forever.
#[allow(clippy::type_complexity)]
pub fn compute_worker_budget(
    mut commands: Commands,
    limits: Res<WorkerLimits>,
    mut budget: ResMut<WorkerBudget>,
    pending: Query<
        (Entity, Option<&DelegatedWorkerTasksParameters>),
        (With<DelegatedWorkerTaskMarker>, Without<Deleted>),
    >,
    delegators: Query<(), With<WorkerTaskMarker>>,
) {
    let mut pendings = 0u32;
    for (e, task) in &pending {
        if let Some(task) = task
            && !delegators.contains(task.delegator_id())
        {
            commands.entity(e).insert(Deleted);
            continue;
        }
        pendings += 1;
    }
    budget.shared = limits
        .max_pendings
        .saturating_sub(pendings)
        .saturating_sub(limits.external_load);
    budget.reserved.clear();
    budget.paused = limits.is_paused();
}

/// Emit the dispatch event for newly delegated tasks, most urgent first: the
/// platform starts the stack in order, so within a frame a terrain mesh
/// (`High`) reaches the worker pool before an MVT parse (`Medium`). Ordering
/// across frames comes from the shared [`WorkerBudget`], which admits at most
/// as many tasks as the pool can run. The matching removal event is emitted
/// by [`remove`] at the despawn point, so a deletion can never slip past the
/// event.
#[allow(clippy::type_complexity)]
pub fn commit(
    mut events: ResMut<EventStore>,
    added: Query<(Entity, &Priority), (Added<DelegatedWorkerTaskMarker>, Without<Deleted>)>,
) {
    for (e, _) in added.iter().sort::<&Priority>() {
        events.worker_task_delegated.push(e);
    }
}

#[allow(clippy::type_complexity)]
pub fn handle_completed_event(
    mut commands: Commands,
    mut loaded_ev: MessageReader<WorkerTaskCompletedEvent>,
    mut buf: ResMut<BufferStore>,
    constructors: Query<
        Entity,
        (
            With<WorkerTaskMarker>,
            Without<WorkerTaskCompleted>,
            Without<Deleted>,
        ),
    >,
) {
    for e in loaded_ev.read() {
        // One body per task kind, differing only in the result type: deliver
        // the result to a live delegator, or — when the task was deleted
        // before completion — free the result's buffers through the same
        // `FreeResultBuffers` impl the component's `on_remove` hook uses, so
        // the set of owned handles cannot drift between the two paths.
        macro_rules! settle {
            ($delegator_id:expr, $value:expr) => {
                if constructors.contains(*$delegator_id) {
                    commands
                        .entity(*$delegator_id)
                        .insert(($value.clone(), WorkerTaskCompleted));
                } else {
                    $value.remove_from_buf(&mut buf);
                }
            };
        }
        match &e.result {
            DelegatedWorkerTasksResult::ConstructTerrainMesh(DelegatedWorkerTask {
                delegator_id,
                value,
            }) => settle!(delegator_id, value),
            DelegatedWorkerTasksResult::UpsampleTerrainMesh(DelegatedWorkerTask {
                delegator_id,
                value,
            }) => settle!(delegator_id, value),
            DelegatedWorkerTasksResult::ConstructPolygonBatchedFeature(DelegatedWorkerTask {
                delegator_id,
                value,
            }) => settle!(delegator_id, value),
            DelegatedWorkerTasksResult::ConstructPolylineBatchedFeature(DelegatedWorkerTask {
                delegator_id,
                value,
            }) => settle!(delegator_id, value),
            DelegatedWorkerTasksResult::ParseMvtTile(DelegatedWorkerTask {
                delegator_id,
                value,
            }) => settle!(delegator_id, value),
        }
        // `try_insert`: a delegatee torn down with its deleted delegator may
        // already be despawned.
        if let Ok(mut delegatee) = commands.get_entity(e.parameters_id) {
            delegatee.try_insert(Deleted);
        }
    }
}

/// Release delegators whose delegated task failed on the platform side (see
/// [`WorkerTaskFailedEvent`]): mark them `Deleted` so [`remove`] tears them
/// down — delegatee included — and their pending dispatch slot frees. A
/// delegator already despawned (cancelled while the failure was in flight)
/// fails the entity lookup and is skipped.
pub fn handle_failed_event(
    mut commands: Commands,
    mut failed_ev: MessageReader<WorkerTaskFailedEvent>,
) {
    for e in failed_ev.read() {
        if let Ok(mut delegator) = commands.get_entity(e.delegator_id) {
            delegator.try_insert(Deleted);
        }
    }
}

/// Tear down deleted worker-task entities. Any unconsumed BufferStore handles
/// held by their result components are freed by the components' `on_remove`
/// hooks (see each task's `component.rs`), so this stays task-agnostic.
///
/// Deletion is propagated to the delegatee and the removal event is emitted
/// HERE, atomically with the despawn, and nowhere else. Doing either in a
/// separate system leaves a window — systems outside the worker chain apply
/// their `Deleted` inserts at arbitrary points between chained systems, so a
/// deletion landing between that system and this despawn would orphan the
/// delegatee (its in-flight worker-pool promise never aborted) or lose the
/// removal event.
#[allow(clippy::type_complexity)]
pub fn remove(
    mut commands: Commands,
    mut events: ResMut<EventStore>,
    constructors: Query<
        (
            Entity,
            Option<&WorkerTaskDelegateeMarker>,
            Has<DelegatedWorkerTaskMarker>,
        ),
        (
            With<Deleted>,
            Or<(With<WorkerTaskMarker>, With<DelegatedWorkerTaskMarker>)>,
        ),
    >,
) {
    for (e, delegatee, is_delegatee) in &constructors {
        if let Some(m) = delegatee
            && let Ok(mut de) = commands.get_entity(m.0)
        {
            // The delegatee is despawned on the next pass, once the Deleted
            // inserted here lands and its own removal event is emitted.
            // `try_insert`: when the delegatee is itself Deleted in this same
            // pass (e.g. a cancel marked both) and was iterated first, its
            // queued despawn applies before this insert; a plain `insert`
            // would panic on the despawned entity.
            de.try_insert(Deleted);
        }
        if is_delegatee {
            events.worker_task_removed.push(e);
        }
        commands.entity(e).despawn();
    }
}

#[cfg(test)]
mod test {
    use bevy_ecs::message::Messages;
    use bevy_ecs::system::RunSystemOnce;
    use bevy_ecs::world::World;
    use navara_component::Deleted;
    use navara_event_store::EventStore;

    use crate::{
        DelegatedWorkerTaskMarker, WorkerTaskDelegateeMarker, WorkerTaskFailedEvent,
        WorkerTaskMarker,
    };

    /// Within one frame the dispatch events come out most urgent first,
    /// whatever order the tasks were spawned in.
    #[test]
    fn it_should_emit_delegated_tasks_most_urgent_first() {
        use navara_component::Priority;

        let mut world = World::new();
        world.init_resource::<EventStore>();
        let medium = world
            .spawn((DelegatedWorkerTaskMarker, Priority::Medium))
            .id();
        let high = world
            .spawn((DelegatedWorkerTaskMarker, Priority::High))
            .id();
        let low = world.spawn((DelegatedWorkerTaskMarker, Priority::Low)).id();

        world.run_system_once(super::commit).unwrap();

        let events = world.resource::<EventStore>();
        assert_eq!(events.worker_task_delegated, vec![high, medium, low]);
    }

    /// The budget is what the limit leaves after the delegatees still in
    /// flight; one already marked for teardown no longer counts.
    #[test]
    fn it_should_budget_the_limit_minus_pending_delegatees() {
        let mut world = World::new();
        world.insert_resource(super::WorkerLimits {
            max_pendings: 3,
            ..Default::default()
        });
        world.init_resource::<super::WorkerBudget>();
        world.spawn(DelegatedWorkerTaskMarker);
        world.spawn(DelegatedWorkerTaskMarker);
        world.spawn((DelegatedWorkerTaskMarker, Deleted));

        world.run_system_once(super::compute_worker_budget).unwrap();
        assert_eq!(world.resource::<super::WorkerBudget>().shared, 1);

        world.spawn(DelegatedWorkerTaskMarker);
        world.spawn(DelegatedWorkerTaskMarker);
        world.run_system_once(super::compute_worker_budget).unwrap();
        assert_eq!(world.resource::<super::WorkerBudget>().shared, 0);
    }

    /// Pool work the frontend runs outside delegated tasks takes from the
    /// same limit.
    #[test]
    fn it_should_budget_the_limit_minus_the_external_load() {
        let mut world = World::new();
        world.insert_resource(super::WorkerLimits {
            max_pendings: 4,
            external_load: 2,
            ..Default::default()
        });
        world.init_resource::<super::WorkerBudget>();
        world.spawn(DelegatedWorkerTaskMarker);

        world.run_system_once(super::compute_worker_budget).unwrap();
        assert_eq!(world.resource::<super::WorkerBudget>().shared, 1);

        world.resource_mut::<super::WorkerLimits>().external_load = 8;
        world.run_system_once(super::compute_worker_budget).unwrap();
        assert_eq!(world.resource::<super::WorkerBudget>().shared, 0);
    }

    /// A kind with waiting tasks and nothing in flight gets a slot reserved
    /// out of the shared budget, ahead of the kinds earlier in the chain; the
    /// total never exceeds the limit.
    #[test]
    fn it_should_reserve_a_slot_for_a_starving_kind_within_the_limit() {
        use crate::construct_terrain_mesh::ConstructTerrainMeshMarker;
        use crate::parse_mvt_tile::ParseMvtTileMarker;
        use crate::upsample_terrain_mesh::UpsampleTerrainMeshMarker;

        let mut world = World::new();
        world.insert_resource(super::WorkerLimits {
            max_pendings: 2,
            ..Default::default()
        });
        world.init_resource::<super::WorkerBudget>();
        // Terrain has one task in flight and more waiting; MVT waits with
        // nothing in flight; upsample has nothing waiting.
        world.spawn((ConstructTerrainMeshMarker, DelegatedWorkerTaskMarker));
        world.spawn((ConstructTerrainMeshMarker, WorkerTaskMarker));
        world.spawn((ParseMvtTileMarker, WorkerTaskMarker));

        world.run_system_once(super::compute_worker_budget).unwrap();
        world
            .run_system_once(super::reserve_for_starving_kind::<ConstructTerrainMeshMarker>)
            .unwrap();
        world
            .run_system_once(super::reserve_for_starving_kind::<UpsampleTerrainMeshMarker>)
            .unwrap();
        world
            .run_system_once(super::reserve_for_starving_kind::<ParseMvtTileMarker>)
            .unwrap();

        let mut budget = world.resource::<super::WorkerBudget>().clone();
        assert_eq!(budget.allowance::<ConstructTerrainMeshMarker>(), 0);
        assert_eq!(budget.allowance::<UpsampleTerrainMeshMarker>(), 0);
        assert_eq!(budget.allowance::<ParseMvtTileMarker>(), 1);
        budget.spend::<ParseMvtTileMarker>(1);
        assert_eq!(budget.allowance::<ParseMvtTileMarker>(), 0);
    }

    /// The reserved slot is spent first; the rest comes out of the shared
    /// budget, and a paused budget (limit 0) admits nothing at all.
    #[test]
    fn it_should_spend_the_reservation_first_and_admit_nothing_when_paused() {
        use crate::construct_terrain_mesh::ConstructTerrainMeshMarker as Terrain;
        use crate::parse_mvt_tile::ParseMvtTileMarker as Mvt;

        let mut budget = super::WorkerBudget {
            shared: 2,
            ..Default::default()
        };
        budget.reserve::<Terrain>();
        assert_eq!(budget.shared, 1);
        assert_eq!(budget.allowance::<Terrain>(), 2);
        assert_eq!(budget.allowance::<Mvt>(), 1);
        budget.spend::<Terrain>(2);
        assert_eq!(budget.allowance::<Terrain>(), 0);
        assert_eq!(budget.allowance::<Mvt>(), 0);

        let mut world = World::new();
        world.insert_resource(super::WorkerLimits {
            intake_paused: true,
            ..Default::default()
        });
        world.init_resource::<super::WorkerBudget>();
        world.run_system_once(super::compute_worker_budget).unwrap();
        let mut paused = world.resource::<super::WorkerBudget>().clone();
        assert!(paused.paused);
        paused.reserve::<Terrain>();
        assert_eq!(paused.allowance::<Terrain>(), 0);
    }

    /// A delegatee whose delegator vanished without the `remove` pass is torn
    /// down instead of holding a budget slot forever.
    #[test]
    fn it_should_not_budget_a_delegatee_whose_delegator_is_gone() {
        use crate::parse_mvt_tile::ParseMvtTileParameters;
        use crate::{DelegatedWorkerTask, DelegatedWorkerTasksParameters};

        let mut world = World::new();
        world.insert_resource(super::WorkerLimits {
            max_pendings: 2,
            ..Default::default()
        });
        world.init_resource::<super::WorkerBudget>();
        let task = |delegator| {
            DelegatedWorkerTasksParameters::ParseMvtTile(DelegatedWorkerTask::new(
                delegator,
                ParseMvtTileParameters {
                    pbf_handle: 0,
                    x: 0,
                    y: 0,
                    z: 0,
                    tile_extent: None,
                    configs: vec![],
                    compression: 0,
                },
            ))
        };
        let live = world.spawn(WorkerTaskMarker).id();
        let gone = world.spawn(WorkerTaskMarker).id();
        world.spawn((DelegatedWorkerTaskMarker, task(live)));
        let orphan = world.spawn((DelegatedWorkerTaskMarker, task(gone))).id();
        world.despawn(gone);

        world.run_system_once(super::compute_worker_budget).unwrap();

        assert_eq!(world.resource::<super::WorkerBudget>().shared, 1);
        assert!(world.entity(orphan).contains::<Deleted>());
    }

    /// A failure report must release the delegator: marking it `Deleted` lets
    /// the `remove` pass tear it down and free its pending dispatch slot.
    #[test]
    fn it_should_release_the_delegator_of_a_failed_task() {
        let mut world = World::new();
        world.init_resource::<Messages<WorkerTaskFailedEvent>>();
        let delegator = world.spawn(WorkerTaskMarker).id();
        world.write_message(WorkerTaskFailedEvent {
            delegator_id: delegator,
        });

        world.run_system_once(super::handle_failed_event).unwrap();

        assert!(world.entity(delegator).contains::<Deleted>());
    }

    /// A failure that raced a cancellation (the engine already despawned the
    /// delegator) must be ignored instead of panicking.
    #[test]
    fn it_should_ignore_a_failure_for_an_already_despawned_delegator() {
        let mut world = World::new();
        world.init_resource::<Messages<WorkerTaskFailedEvent>>();
        let delegator = world.spawn(WorkerTaskMarker).id();
        world.despawn(delegator);
        world.write_message(WorkerTaskFailedEvent {
            delegator_id: delegator,
        });

        world.run_system_once(super::handle_failed_event).unwrap();
    }

    /// Deleting a delegator must propagate to its delegatee at the despawn
    /// point, then emit the delegatee's removal event (so the JS worker pool
    /// aborts the in-flight promise) before the delegatee is despawned on the
    /// next pass.
    #[test]
    fn it_should_tear_down_the_delegatee_and_emit_its_removal_event() {
        let mut world = World::new();
        world.insert_resource(EventStore::default());
        let delegatee = world.spawn(DelegatedWorkerTaskMarker).id();
        let delegator = world
            .spawn((
                WorkerTaskMarker,
                WorkerTaskDelegateeMarker(delegatee),
                Deleted,
            ))
            .id();

        world.run_system_once(super::remove).unwrap();

        assert!(world.get_entity(delegator).is_err());
        assert!(world.entity(delegatee).contains::<Deleted>());
        // The delegator is not a delegated task; no removal event for it.
        assert!(
            world
                .resource::<EventStore>()
                .worker_task_removed
                .is_empty()
        );

        world.run_system_once(super::remove).unwrap();

        assert!(world.get_entity(delegatee).is_err());
        assert_eq!(
            world.resource::<EventStore>().worker_task_removed,
            vec![delegatee]
        );
    }

    /// When a cancel marks the delegator AND the delegatee `Deleted` in the
    /// same frame (as `cancel_evicted_parse_tasks` does), one `remove` pass
    /// handles both. The delegatee's archetype is created first so it is
    /// iterated first: its queued despawn applies before the delegator's
    /// Deleted propagation, which must therefore tolerate the already
    /// despawned target instead of panicking.
    #[test]
    fn it_should_survive_a_delegator_and_delegatee_deleted_in_the_same_pass() {
        let mut world = World::new();
        world.insert_resource(EventStore::default());
        let delegatee = world.spawn((DelegatedWorkerTaskMarker, Deleted)).id();
        let delegator = world
            .spawn((
                WorkerTaskMarker,
                WorkerTaskDelegateeMarker(delegatee),
                Deleted,
            ))
            .id();

        world.run_system_once(super::remove).unwrap();

        assert!(world.get_entity(delegator).is_err());
        assert!(world.get_entity(delegatee).is_err());
        assert_eq!(
            world.resource::<EventStore>().worker_task_removed,
            vec![delegatee]
        );
    }

    /// A delegator whose delegatee is already gone (e.g. torn down by
    /// `handle_completed_event` after a normal completion) must still despawn
    /// cleanly.
    #[test]
    fn it_should_despawn_a_delegator_whose_delegatee_is_already_gone() {
        let mut world = World::new();
        world.insert_resource(EventStore::default());
        let delegatee = world.spawn(DelegatedWorkerTaskMarker).id();
        world.despawn(delegatee);
        let delegator = world
            .spawn((
                WorkerTaskMarker,
                WorkerTaskDelegateeMarker(delegatee),
                Deleted,
            ))
            .id();

        world.run_system_once(super::remove).unwrap();

        assert!(world.get_entity(delegator).is_err());
    }
}
