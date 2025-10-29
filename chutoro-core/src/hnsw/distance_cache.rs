use std::{
    num::NonZeroUsize,
    sync::Mutex,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use lru::LruCache;
use tracing::instrument;

use crate::{datasource::MetricDescriptor, hnsw::error::HnswError};

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
#[derive(Clone, Debug)]
pub struct DistanceCacheConfig {
    max_entries: NonZeroUsize,
    ttl: Option<Duration>,
}

impl DistanceCacheConfig {
    /// Default maximum number of cached distances retained before eviction.
    pub const DEFAULT_MAX_ENTRIES: usize = 1_048_576;

    /// Builds a configuration with the provided maximum capacity.
    pub fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            max_entries,
            ttl: None,
        }
    }

    /// Sets an optional time-to-live applied to cached entries.
    pub fn with_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.ttl = ttl;
        self
    }

    /// Returns the maximum number of cached distances retained before eviction.
    pub fn max_entries(&self) -> NonZeroUsize {
        self.max_entries
    }

    /// Returns the configured time-to-live, if any.
    pub fn ttl(&self) -> Option<Duration> {
        self.ttl
    }
}

impl Default for DistanceCacheConfig {
    fn default() -> Self {
        let max_entries = NonZeroUsize::new(Self::DEFAULT_MAX_ENTRIES)
            .expect("default cache size must be non-zero");
        Self::new(max_entries)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct DistanceKey {
    metric: MetricDescriptor,
    left: usize,
    right: usize,
}

impl DistanceKey {
    fn new(metric: MetricDescriptor, a: usize, b: usize) -> Self {
        let (left, right) = if a <= b { (a, b) } else { (b, a) };
        Self {
            metric,
            left,
            right,
        }
    }
}

#[derive(Clone, Debug)]
struct CacheEntry {
    value: f32,
    inserted: Instant,
}

#[derive(Debug)]
pub(crate) struct PendingMiss {
    key: DistanceKey,
    started: Instant,
    left: usize,
    right: usize,
}

#[derive(Debug)]
pub(crate) enum LookupOutcome {
    Hit(f32),
    Miss(PendingMiss),
}

#[derive(Debug)]
pub(crate) struct DistanceCache {
    entries: DashMap<DistanceKey, CacheEntry>,
    usage: Mutex<LruCache<DistanceKey, ()>>,
    config: DistanceCacheConfig,
}

impl DistanceCache {
    pub(crate) fn new(config: DistanceCacheConfig) -> Self {
        let capacity = config.max_entries();
        let cap_usize = capacity.get();
        Self {
            entries: DashMap::with_capacity(cap_usize),
            usage: Mutex::new(LruCache::new(capacity)),
            config,
        }
    }

    #[instrument(level = "trace", skip(self, metric))]
    pub(crate) fn begin_lookup(
        &self,
        metric: &MetricDescriptor,
        left: usize,
        right: usize,
    ) -> LookupOutcome {
        let started = Instant::now();
        let key = DistanceKey::new(metric.clone(), left, right);
        if let Some(entry) = self.entries.get(&key) {
            if self.is_expired(&entry) {
                drop(entry);
                self.entries.remove(&key);
                self.remove_from_usage(&key);
                self.record_eviction();
                self.record_miss();
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
            self.record_hit(started.elapsed());
            LookupOutcome::Hit(value)
        } else {
            self.record_miss();
            LookupOutcome::Miss(PendingMiss {
                key,
                started,
                left,
                right,
            })
        }
    }

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
        self.entries.insert(
            key.clone(),
            CacheEntry {
                value,
                inserted: Instant::now(),
            },
        );
        self.touch(&key);
        self.record_lookup_latency(started.elapsed());
        Ok(value)
    }

    fn is_expired(&self, entry: &CacheEntry) -> bool {
        self.config
            .ttl()
            .is_some_and(|ttl| entry.inserted.elapsed() > ttl)
    }

    fn touch(&self, key: &DistanceKey) {
        let mut usage = self
            .usage
            .lock()
            .expect("distance cache usage mutex poisoned");
        if let Some((evicted, _)) = usage.push(key.clone(), ()) {
            self.entries.remove(&evicted);
            self.record_eviction();
        }
    }

    fn remove_from_usage(&self, key: &DistanceKey) {
        if let Ok(mut usage) = self.usage.lock() {
            usage.pop(key);
        }
    }

    #[cfg(feature = "metrics")]
    fn record_hit(&self, elapsed: Duration) {
        metrics::counter!("distance_cache_hits").increment(1);
        metrics::histogram!("distance_cache_lookup_latency_histogram")
            .record(elapsed.as_secs_f64());
    }

    #[cfg(not(feature = "metrics"))]
    fn record_hit(&self, _elapsed: Duration) {}

    #[cfg(feature = "metrics")]
    fn record_miss(&self) {
        metrics::counter!("distance_cache_misses").increment(1);
    }

    #[cfg(not(feature = "metrics"))]
    fn record_miss(&self) {}

    #[cfg(feature = "metrics")]
    fn record_eviction(&self) {
        metrics::counter!("distance_cache_evictions").increment(1);
    }

    #[cfg(not(feature = "metrics"))]
    fn record_eviction(&self) {}

    #[cfg(feature = "metrics")]
    fn record_lookup_latency(&self, elapsed: Duration) {
        metrics::histogram!("distance_cache_lookup_latency_histogram")
            .record(elapsed.as_secs_f64());
    }

    #[cfg(not(feature = "metrics"))]
    fn record_lookup_latency(&self, _elapsed: Duration) {}
}

impl PendingMiss {}
