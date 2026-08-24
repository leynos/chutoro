//! Concurrent distance cache with sharded LRU bookkeeping for HNSW.
//!
//! Avoids recomputing distances across threads, exposes cache metrics, and
//! enforces deterministic eviction even under high contention by hashing keys
//! into fixed-capacity shards.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    sync::Mutex,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use lru::LruCache;
use tracing::instrument;

use crate::{datasource::MetricDescriptor, hnsw::error::HnswError};

mod metric_hooks;

/// Configuration parameters for the distance cache used by [`crate::CpuHnsw`].
///
/// # Examples
/// ```
/// use chutoro_core::DistanceCacheConfig;
/// use std::num::NonZeroUsize;
///
/// let config = DistanceCacheConfig::new(NonZeroUsize::new(1024).unwrap())
///     .with_ttl(None);
/// assert_eq!(config.max_entries().get(), 1024);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceCacheConfig {
    /// Maximum number of distances retained across all shards.
    max_entries: NonZeroUsize,
    /// Optional age after which a cache entry expires.
    ttl: Option<Duration>,
}

impl DistanceCacheConfig {
    /// Default maximum number of cached distances retained before eviction.
    pub const DEFAULT_MAX_ENTRIES: usize = 1_048_576;

    /// Builds a configuration with the provided maximum capacity.
    #[must_use]
    pub const fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            max_entries,
            ttl: None,
        }
    }

    /// Sets an optional time-to-live applied to cached entries.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.ttl = ttl;
        self
    }

    /// Updates the maximum number of cached entries retained before eviction.
    ///
    /// # Examples
    /// ```rust
    /// use chutoro_core::DistanceCacheConfig;
    /// use std::num::NonZeroUsize;
    ///
    /// let config = DistanceCacheConfig::default()
    ///     .with_max_entries(NonZeroUsize::new(2).unwrap());
    /// assert_eq!(config.max_entries().get(), 2);
    /// ```
    #[must_use]
    pub const fn with_max_entries(mut self, max: NonZeroUsize) -> Self {
        self.max_entries = max;
        self
    }

    /// Returns the maximum number of cached distances retained before eviction.
    #[must_use]
    pub const fn max_entries(&self) -> NonZeroUsize {
        self.max_entries
    }

    /// Returns the configured time-to-live, if any.
    #[must_use]
    pub const fn ttl(&self) -> Option<Duration> {
        self.ttl
    }
}

impl Default for DistanceCacheConfig {
    fn default() -> Self {
        let max_entries = NonZeroUsize::new(Self::DEFAULT_MAX_ENTRIES).unwrap_or(NonZeroUsize::MIN);
        Self::new(max_entries)
    }
}

/// Canonical cache key for an unordered pair of nodes and a metric.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct DistanceKey {
    /// Metric used to compute the cached distance.
    metric: MetricDescriptor,
    /// Lower node identifier in the canonical pair.
    left: usize,
    /// Higher node identifier in the canonical pair.
    right: usize,
}

impl DistanceKey {
    /// Construct a key by canonicalising the node-pair order.
    const fn new(metric: MetricDescriptor, a: usize, b: usize) -> Self {
        let (left, right) = if a <= b { (a, b) } else { (b, a) };
        Self {
            metric,
            left,
            right,
        }
    }
}

/// Cached finite distance and the instant at which it was inserted.
#[derive(Clone, Debug)]
struct CacheEntry {
    /// Distance value retained for a key.
    value: f32,
    /// Insertion instant used for time-to-live expiry.
    inserted: Instant,
}

/// Metadata retained while an uncached distance is computed.
#[derive(Debug)]
pub(crate) struct PendingMiss {
    /// Cache key to populate once the distance is computed.
    key: DistanceKey,
    /// Lookup start time used to record latency.
    started: Instant,
    /// First node identifier for non-finite-distance errors.
    left: usize,
    /// Second node identifier for non-finite-distance errors.
    right: usize,
}

/// Result of looking up a distance before computing a miss.
#[derive(Debug)]
pub(crate) enum LookupOutcome {
    /// Cached distance available for immediate reuse.
    Hit(f32),
    /// Cache metadata for a distance that must be computed.
    Miss(PendingMiss),
}

/// Upper bound on LRU bookkeeping shards.
const DEFAULT_LRU_SHARDS: usize = 64;
/// Desired number of entries assigned to each LRU shard.
const TARGET_LRU_ENTRIES_PER_SHARD: usize = 4096;

/// LRU bookkeeping for the subset of keys assigned to one shard.
#[derive(Debug)]
struct LruShard {
    /// Usage order used to select the least-recently-used key for eviction.
    usage: Mutex<LruCache<DistanceKey, ()>>,
}

impl LruShard {
    /// Allocate an empty shard with a non-zero key capacity.
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            usage: Mutex::new(LruCache::new(capacity)),
        }
    }
}

#[derive(Debug)]
pub(crate) struct DistanceCache {
    /// Concurrent distance values indexed by their canonical keys.
    entries: DashMap<DistanceKey, CacheEntry>,
    /// Sharded LRU bookkeeping aligned with the cached keys.
    shards: Vec<LruShard>,
    /// Capacity and expiry policy applied to this cache.
    config: DistanceCacheConfig,
}

impl DistanceCache {
    /// Builds a cache using the supplied configuration for capacity and
    /// optional time-to-live limits.
    ///
    /// Entries are evicted when the configured maximum is exceeded or their
    /// time-to-live expires.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use chutoro_core::DistanceCacheConfig;
    /// use crate::hnsw::distance_cache::DistanceCache;
    /// use std::num::NonZeroUsize;
    ///
    /// let config = DistanceCacheConfig::new(NonZeroUsize::new(4).unwrap());
    /// let cache = DistanceCache::new(config.clone());
    /// assert_eq!(config.max_entries().get(), 4);
    /// let _ = cache;
    /// ```
    /// Build an empty cache from the supplied capacity and expiry policy.
    pub(crate) fn new(config: DistanceCacheConfig) -> Self {
        let capacity = config.max_entries();
        let cap_usize = capacity.get();
        let shard_capacities = lru_shard_capacities(cap_usize);
        let shards = shard_capacities.into_iter().map(LruShard::new).collect();
        Self {
            entries: DashMap::with_capacity(cap_usize),
            shards,
            config,
        }
    }

    /// Return a cached distance or metadata for completing a miss.
    #[instrument(level = "trace", skip(self, metric))]
    pub(crate) fn begin_lookup(
        &self,
        metric: &MetricDescriptor,
        left: usize,
        right: usize,
    ) -> LookupOutcome {
        let started = Instant::now();
        let key = DistanceKey::new(metric.clone(), left, right);
        if self.shards.is_empty() {
            tracing::error!("distance cache has no LRU shards; bypassing cache");
            metric_hooks::record_miss();
            return LookupOutcome::Miss(PendingMiss {
                key,
                started,
                left,
                right,
            });
        }
        if let Some(entry) = self.entries.get(&key) {
            if self.is_expired(&entry) {
                drop(entry);
                self.entries.remove(&key);
                self.remove_from_usage(&key);
                metric_hooks::record_eviction();
                metric_hooks::record_miss();
                return LookupOutcome::Miss(PendingMiss {
                    key,
                    started,
                    left,
                    right,
                });
            }
            let value = entry.value;
            drop(entry);
            self.touch(&key);
            metric_hooks::record_hit(started.elapsed());
            LookupOutcome::Hit(value)
        } else {
            metric_hooks::record_miss();
            LookupOutcome::Miss(PendingMiss {
                key,
                started,
                left,
                right,
            })
        }
    }

    /// Validate and store a computed miss, returning its finite distance.
    pub(crate) fn complete_miss(&self, miss: PendingMiss, value: f32) -> Result<f32, HnswError> {
        let PendingMiss {
            key,
            started,
            left,
            right,
        } = miss;
        if !value.is_finite() {
            tracing::warn!(
                ?key,
                %value,
                "rejecting non-finite distance from cache lookup"
            );
            return Err(HnswError::NonFiniteDistance { left, right });
        }
        if self.shards.is_empty() {
            metric_hooks::record_lookup_latency(started.elapsed());
            return Ok(value);
        }
        self.entries.insert(
            key.clone(),
            CacheEntry {
                value,
                inserted: Instant::now(),
            },
        );
        self.touch(&key);
        metric_hooks::record_lookup_latency(started.elapsed());
        Ok(value)
    }

    /// Report whether an entry exceeds the configured time-to-live.
    fn is_expired(&self, entry: &CacheEntry) -> bool {
        self.config
            .ttl()
            .is_some_and(|ttl| entry.inserted.elapsed() > ttl)
    }

    /// Mark a cache key as recently used and evict an LRU key when needed.
    fn touch(&self, key: &DistanceKey) {
        let Some(shard) = self.shard_for_key(key) else {
            return;
        };
        // Recover from a poisoned lock: the LRU usage list stays coherent
        // because each mutation below is applied atomically under the guard.
        let mut usage = shard
            .usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((evicted, ())) = usage.push(key.clone(), ()) {
            self.entries.remove(&evicted);
            metric_hooks::record_eviction();
        }
    }

    /// Remove a key from LRU usage while preserving a concurrently restored key.
    fn remove_from_usage(&self, key: &DistanceKey) {
        let Some(shard) = self.shard_for_key(key) else {
            return;
        };
        // Recover from a poisoned lock: the LRU usage list stays coherent
        // because each mutation below is applied atomically under the guard.
        let mut usage = shard
            .usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(evicted) = self.try_restore_and_get_evicted(&mut usage, key) {
            self.entries.remove(&evicted);
            metric_hooks::record_eviction();
        }
    }

    /// Restore a key still present in the value map and return any eviction.
    fn try_restore_and_get_evicted(
        &self,
        usage: &mut LruCache<DistanceKey, ()>,
        key: &DistanceKey,
    ) -> Option<DistanceKey> {
        let was_in_usage = usage.pop(key).is_some();
        if !was_in_usage {
            return None;
        }

        let should_restore = self.entries.contains_key(key);
        if !should_restore {
            return None;
        }

        let restored = usage.push(key.clone(), ());
        restored.map(|(evicted, ())| evicted)
    }

    /// Return the LRU shard deterministically assigned to a cache key.
    fn shard_for_key(&self, key: &DistanceKey) -> Option<&LruShard> {
        let shard_count = self.shards.len();
        if shard_count == 0 {
            return None;
        }
        let index = if shard_count == 1 {
            0
        } else {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            let shard_count_as_u64 = u64::try_from(shard_count).ok()?;
            let hash_remainder = hasher.finish().checked_rem(shard_count_as_u64)?;
            usize::try_from(hash_remainder).ok()?
        };
        self.shards.get(index)
    }
}

// no inherent methods on PendingMiss

/// Divide total capacity into bounded, non-zero LRU shard capacities.
fn lru_shard_capacities(total_capacity: usize) -> Vec<NonZeroUsize> {
    debug_assert!(total_capacity > 0, "total capacity must be non-zero");
    let desired_shards = total_capacity.div_ceil(TARGET_LRU_ENTRIES_PER_SHARD);
    let shard_count = desired_shards
        .clamp(1, DEFAULT_LRU_SHARDS)
        .min(total_capacity);
    let base = total_capacity.checked_div(shard_count).unwrap_or_default();
    let remainder = total_capacity.checked_rem(shard_count).unwrap_or_default();

    (0..shard_count)
        .map(|index| {
            let extra = usize::from(index < remainder);
            let shard_capacity = base + extra;
            NonZeroUsize::new(shard_capacity).unwrap_or(NonZeroUsize::MIN)
        })
        .collect()
}
