//! Tile data manager for deduplicating data requests based on URL + byte range.
//!
//! When multiple layers (terrain, hillshade, etc.) request the same bytes,
//! the DataManager ensures only one network request is made and the
//! data is shared via BufferStore handles with reference counting.

use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::Resource;
use navara_buffer_store::{BufferStore, Handle};
use rustc_hash::{FxHashMap, FxHashSet};

/// Identity of a shared fetch: a URL plus an optional HTTP byte range.
///
/// The byte range is part of the key because a `Range: bytes=offset-…` GET
/// returns a *different payload* than a full GET — or than a different range —
/// of the same URL. Deduplicating on URL alone would let two consumers asking
/// for distinct ranges share one handle/fetch, handing at least one of them the
/// wrong bytes. `None` (full-resource GET) is its own distinct key.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct RequestKey {
    url: String,
    byte_range: Option<(u64, u64)>,
}

/// Manages resource deduplication and sharing across multiple consumers.
///
/// The DataManager tracks which (URL, byte range) pairs have been requested and
/// maintains reference counts to ensure resources are only loaded once and
/// cleaned up when no longer needed.
#[derive(Default, Resource)]
pub struct DataManager {
    /// Maps request key (URL + byte range) to resource entry (handle + consumers)
    url_registry: FxHashMap<RequestKey, ResourceEntry>,
    /// Maps entity to its request key for cleanup lookup
    entity_to_url: FxHashMap<Entity, RequestKey>,
    /// Keys whose fetch leader was removed while other consumers remained;
    /// drained by [`DataManager::promote_orphaned_leaders`].
    orphaned: FxHashSet<RequestKey>,
}

/// Information about a shared resource.
pub struct ResourceEntry {
    /// BufferStore handle for the shared data
    pub handle: Handle,
    /// Entities currently sharing the handle.
    pub consumers: FxHashSet<Entity>,
    /// The consumer whose entity the fetch was dispatched for, while that
    /// fetch is in flight. Other consumers wait on the same handle; when the
    /// leader is removed one of them is promoted (see
    /// [`DataManager::promote_orphaned_leaders`]). `None` once the fetch
    /// settled or failed, so a new consumer can dispatch again.
    pub leader: Option<Entity>,
}

/// Outcome of [`DataManager::unregister_consumer`].
#[derive(Debug, PartialEq, Eq)]
pub struct Unregistered {
    pub url: String,
    pub handle: Handle,
    /// This was the last consumer: the shared buffer can be freed.
    pub last_consumer: bool,
}

impl DataManager {
    /// Create a new DataManager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a consumer for a URL + byte range.
    ///
    /// Pass the same `byte_range` that the consumer's `DataRequester` carries
    /// (`None` for a full-resource GET). Consumers sharing a handle/fetch must
    /// agree on the bytes they want; see [`RequestKey`].
    ///
    /// Returns:
    /// - `handle`: BufferStore handle for the resource (new or existing)
    /// - `is_new_request`: true if this is the first request for this key
    /// - `fetch_already_enqueued`: true if a fetch has already been triggered for this key
    ///
    /// If the key is already registered, the existing handle is returned and
    /// the reference count is incremented. Otherwise, a new handle is created.
    pub fn register_consumer(
        &mut self,
        url: String,
        byte_range: Option<(u64, u64)>,
        entity: Entity,
        buf: &mut BufferStore,
    ) -> (Handle, bool, bool) {
        let key = RequestKey { url, byte_range };
        let is_new = !self.url_registry.contains_key(&key);

        let entry = self
            .url_registry
            .entry(key.clone())
            .or_insert_with(|| ResourceEntry {
                handle: buf.new_handle(),
                consumers: FxHashSet::default(),
                leader: None,
            });

        let fetch_already_enqueued = entry.leader.is_some();
        entry.consumers.insert(entity);
        self.entity_to_url.insert(entity, key);

        (entry.handle, is_new, fetch_already_enqueued)
    }

    /// Unregister a consumer; `None` if the entity was not registered.
    pub fn unregister_consumer(&mut self, entity: Entity) -> Option<Unregistered> {
        let key = self.entity_to_url.remove(&entity)?;
        let entry = self.url_registry.get_mut(&key)?;

        entry.consumers.remove(&entity);
        let handle = entry.handle;

        if entry.consumers.is_empty() {
            self.url_registry.remove(&key);
            self.orphaned.remove(&key);
            return Some(Unregistered {
                url: key.url,
                handle,
                last_consumer: true,
            });
        }

        if entry.leader == Some(entity) {
            entry.leader = None;
            self.orphaned.insert(key.clone());
        }
        Some(Unregistered {
            url: key.url,
            handle,
            last_consumer: false,
        })
    }

    /// Re-elect a leader for every key whose leader was removed while other
    /// consumers remained, and return the new leaders: each must dispatch the
    /// fetch again. `urgency` is `None` for a consumer that is not eligible
    /// (gone, or no longer pending on the shared fetch); among the eligible,
    /// the smallest urgency (most urgent) wins. A key with no eligible
    /// consumer is dropped, since a later consumer finds no leader and
    /// dispatches by itself.
    pub fn promote_orphaned_leaders<K: Ord>(
        &mut self,
        mut urgency: impl FnMut(Entity) -> Option<K>,
    ) -> Vec<Entity> {
        let mut promoted = Vec::new();
        for key in self.orphaned.drain() {
            let Some(entry) = self.url_registry.get_mut(&key) else {
                continue;
            };
            if entry.leader.is_some() {
                continue;
            }
            let mut next: Option<(K, Entity)> = None;
            for &consumer in entry.consumers.iter() {
                let Some(u) = urgency(consumer) else {
                    continue;
                };
                if next.as_ref().is_none_or(|(best, _)| u < *best) {
                    next = Some((u, consumer));
                }
            }
            if let Some((_, next)) = next {
                entry.leader = Some(next);
                promoted.push(next);
            }
        }
        promoted
    }

    /// Get the reference count for a URL + byte range.
    pub fn get_ref_count(&self, url: &str, byte_range: Option<(u64, u64)>) -> usize {
        let key = RequestKey {
            url: url.to_string(),
            byte_range,
        };
        self.url_registry
            .get(&key)
            .map(|e| e.consumers.len())
            .unwrap_or(0)
    }

    /// Get total number of registered URLs.
    pub fn len(&self) -> usize {
        self.url_registry.len()
    }

    /// Check if there are no registered URLs.
    pub fn is_empty(&self) -> bool {
        self.url_registry.is_empty()
    }

    /// Make `entity` the fetch leader of its URL. Returns true if the caller
    /// should dispatch the fetch, false if another consumer already leads it.
    ///
    /// For entities not registered in DataManager (unmanaged DataRequesters),
    /// returns true to allow the event.
    pub fn try_mark_fetch_enqueued(&mut self, entity: Entity) -> bool {
        if let Some(entry) = self
            .entity_to_url
            .get(&entity)
            .and_then(|key| self.url_registry.get_mut(key))
        {
            if entry.leader.is_some() {
                return false;
            }
            entry.leader = Some(entity);
            return true;
        }
        true
    }

    /// Clear the fetch leader of an entity's URL once its fetch succeeded or
    /// failed, so a future consumer can dispatch again.
    pub fn reset_fetch_enqueued(&mut self, entity: Entity) {
        if let Some(entry) = self
            .entity_to_url
            .get(&entity)
            .and_then(|key| self.url_registry.get_mut(key))
        {
            entry.leader = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;

    #[test]
    fn test_register_consumer_creates_new_entry() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();
        let (_handle, is_new, fetch_already_enqueued) =
            manager.register_consumer(url.clone(), None, entity, &mut buf);

        assert!(is_new);
        assert!(!fetch_already_enqueued);
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.get_ref_count(&url, None), 1);
    }

    #[test]
    fn test_register_consumer_reuses_existing_handle() {
        let mut world = World::new();
        let entity1 = world.spawn_empty().id();
        let entity2 = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();

        // First registration
        let (handle1, is_new1, fetch_already_enqueued1) =
            manager.register_consumer(url.clone(), None, entity1, &mut buf);
        assert!(is_new1);
        assert!(!fetch_already_enqueued1);

        // Second registration (same URL)
        let (handle2, is_new2, fetch_already_enqueued2) =
            manager.register_consumer(url.clone(), None, entity2, &mut buf);
        assert!(!is_new2);
        assert!(!fetch_already_enqueued2); // First consumer hasn't marked it yet
        assert_eq!(handle1, handle2); // Same handle
        assert_eq!(manager.get_ref_count(&url, None), 2);
        assert_eq!(manager.len(), 1); // Still only 1 URL
    }

    #[test]
    fn test_unregister_consumer_decrements_refcount() {
        let mut world = World::new();
        let entity1 = world.spawn_empty().id();
        let entity2 = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();

        let _ = manager.register_consumer(url.clone(), None, entity1, &mut buf);
        let _ = manager.register_consumer(url.clone(), None, entity2, &mut buf);

        // Unregister first consumer
        let result = manager.unregister_consumer(entity1).unwrap();
        assert_eq!(result.url, url);
        assert!(!result.last_consumer); // Still has one consumer
        assert_eq!(manager.get_ref_count(&url, None), 1);
        // No fetch was led, so nobody is promoted.
        assert!(manager.promote_orphaned_leaders(|_| Some(0)).is_empty());
    }

    #[test]
    fn test_unregister_last_consumer_deletes_entry() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();
        let (handle, _, _) = manager.register_consumer(url.clone(), None, entity, &mut buf);

        // Unregister only consumer
        let result = manager.unregister_consumer(entity).unwrap();
        assert_eq!(result.url, url);
        assert_eq!(result.handle, handle);
        assert!(result.last_consumer);
        assert_eq!(manager.len(), 0); // Entry removed
    }

    #[test]
    fn test_unregister_leader_promotes_a_waiting_follower() {
        let mut world = World::new();
        let leader = world.spawn_empty().id();
        let gone = world.spawn_empty().id();
        let follower = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();
        let _ = manager.register_consumer(url.clone(), None, leader, &mut buf);
        assert!(manager.try_mark_fetch_enqueued(leader));
        let (_, _, enqueued) = manager.register_consumer(url.clone(), None, gone, &mut buf);
        assert!(enqueued);
        let _ = manager.register_consumer(url.clone(), None, follower, &mut buf);
        assert!(!manager.try_mark_fetch_enqueued(follower));

        let result = manager.unregister_consumer(leader).unwrap();
        assert!(!result.last_consumer);

        // A consumer that is no longer waiting (e.g. being despawned) is
        // never chosen, whatever the set's iteration order.
        assert_eq!(
            manager.promote_orphaned_leaders(|e| (e == follower).then_some(0)),
            vec![follower]
        );
        assert!(!manager.try_mark_fetch_enqueued(gone));
        // Drained: a second pass promotes nobody.
        assert!(manager.promote_orphaned_leaders(|_| Some(0)).is_empty());
    }

    #[test]
    fn test_promotion_prefers_the_most_urgent_consumer() {
        let mut world = World::new();
        let leader = world.spawn_empty().id();
        let urgent = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();
        let _ = manager.register_consumer(url.clone(), None, leader, &mut buf);
        assert!(manager.try_mark_fetch_enqueued(leader));
        let mut lazy = Vec::new();
        for _ in 0..8 {
            let e = world.spawn_empty().id();
            let _ = manager.register_consumer(url.clone(), None, e, &mut buf);
            lazy.push(e);
        }
        let _ = manager.register_consumer(url.clone(), None, urgent, &mut buf);
        let _ = manager.unregister_consumer(leader);

        assert_eq!(
            manager.promote_orphaned_leaders(|e| Some(if e == urgent { 0 } else { 4 })),
            vec![urgent]
        );
    }

    #[test]
    fn test_orphan_without_waiting_consumer_is_dropped() {
        let mut world = World::new();
        let leader = world.spawn_empty().id();
        let settled = world.spawn_empty().id();
        let next = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();
        let _ = manager.register_consumer(url.clone(), None, leader, &mut buf);
        assert!(manager.try_mark_fetch_enqueued(leader));
        let _ = manager.register_consumer(url.clone(), None, settled, &mut buf);
        let _ = manager.unregister_consumer(leader);

        assert!(manager.promote_orphaned_leaders(|_| None::<u8>).is_empty());
        // The key is leaderless, so a later consumer dispatches by itself.
        let (_, _, enqueued) = manager.register_consumer(url.clone(), None, next, &mut buf);
        assert!(!enqueued);
        assert!(manager.try_mark_fetch_enqueued(next));
    }

    #[test]
    fn test_orphan_with_new_leader_is_not_promoted_twice() {
        let mut world = World::new();
        let leader = world.spawn_empty().id();
        let follower = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();
        let _ = manager.register_consumer(url.clone(), None, leader, &mut buf);
        assert!(manager.try_mark_fetch_enqueued(leader));
        let _ = manager.register_consumer(url.clone(), None, follower, &mut buf);
        let _ = manager.unregister_consumer(leader);

        // A newly sent consumer took the lead before promotion ran.
        assert!(manager.try_mark_fetch_enqueued(follower));
        assert!(manager.promote_orphaned_leaders(|_| Some(0)).is_empty());
    }

    #[test]
    fn test_reset_clears_the_leader() {
        let mut world = World::new();
        let leader = world.spawn_empty().id();
        let next = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/tile.png".to_string();
        let _ = manager.register_consumer(url.clone(), None, leader, &mut buf);
        assert!(manager.try_mark_fetch_enqueued(leader));
        manager.reset_fetch_enqueued(leader);
        let (_, _, enqueued) = manager.register_consumer(url.clone(), None, next, &mut buf);
        assert!(!enqueued);
        assert!(manager.try_mark_fetch_enqueued(next));
    }

    #[test]
    fn test_multiple_urls() {
        let mut world = World::new();
        let entity1 = world.spawn_empty().id();
        let entity2 = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url1 = "https://example.com/tile1.png".to_string();
        let url2 = "https://example.com/tile2.png".to_string();

        let (handle1, _, _) = manager.register_consumer(url1.clone(), None, entity1, &mut buf);
        let (handle2, _, _) = manager.register_consumer(url2.clone(), None, entity2, &mut buf);

        assert_ne!(handle1, handle2); // Different handles
        assert_eq!(manager.len(), 2);
        assert_eq!(manager.get_ref_count(&url1, None), 1);
        assert_eq!(manager.get_ref_count(&url2, None), 1);
    }

    #[test]
    fn test_same_url_distinct_ranges_do_not_share_handle() {
        let mut world = World::new();
        let entity1 = world.spawn_empty().id();
        let entity2 = world.spawn_empty().id();
        let entity_full = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/archive.pmtiles".to_string();

        // Two consumers want different byte ranges of the same URL, plus one
        // wants the whole resource. All three are distinct fetches and must get
        // distinct handles — otherwise the shared fetch returns the wrong bytes.
        let (handle_a, is_new_a, _) =
            manager.register_consumer(url.clone(), Some((0, 100)), entity1, &mut buf);
        let (handle_b, is_new_b, _) =
            manager.register_consumer(url.clone(), Some((100, 50)), entity2, &mut buf);
        let (handle_full, is_new_full, _) =
            manager.register_consumer(url.clone(), None, entity_full, &mut buf);

        assert!(is_new_a);
        assert!(is_new_b);
        assert!(is_new_full);
        assert_ne!(handle_a, handle_b);
        assert_ne!(handle_a, handle_full);
        assert_ne!(handle_b, handle_full);

        // Three distinct keys for the same URL.
        assert_eq!(manager.len(), 3);
        assert_eq!(manager.get_ref_count(&url, Some((0, 100))), 1);
        assert_eq!(manager.get_ref_count(&url, Some((100, 50))), 1);
        assert_eq!(manager.get_ref_count(&url, None), 1);
    }

    #[test]
    fn test_same_url_same_range_shares_handle() {
        let mut world = World::new();
        let entity1 = world.spawn_empty().id();
        let entity2 = world.spawn_empty().id();
        let mut buf = BufferStore::default();
        let mut manager = DataManager::new();

        let url = "https://example.com/archive.pmtiles".to_string();
        let range = Some((0, 100));

        // Consumers asking for the *same* bytes still dedup onto one handle.
        let (handle1, is_new1, _) =
            manager.register_consumer(url.clone(), range, entity1, &mut buf);
        let (handle2, is_new2, _) =
            manager.register_consumer(url.clone(), range, entity2, &mut buf);

        assert!(is_new1);
        assert!(!is_new2);
        assert_eq!(handle1, handle2);
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.get_ref_count(&url, range), 2);
    }

    #[test]
    fn test_unregister_nonexistent_entity() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let mut manager = DataManager::new();

        let result = manager.unregister_consumer(entity);
        assert!(result.is_none());
    }
}
