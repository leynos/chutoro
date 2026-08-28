//! Metric emission hooks owned exclusively by the HNSW distance cache.

use std::time::Duration;

/// Record a cache hit and its lookup latency.
#[cfg(feature = "metrics")]
pub(super) fn record_hit(elapsed: Duration) {
    metrics::counter!("distance_cache_hits").increment(1);
    metrics::histogram!("distance_cache_lookup_latency_histogram").record(elapsed.as_secs_f64());
}

/// Discard a hit metric when metrics are not compiled.
#[cfg(not(feature = "metrics"))]
pub(super) const fn record_hit(_elapsed: Duration) {}

/// Record a cache miss.
#[cfg(feature = "metrics")]
pub(super) fn record_miss() {
    metrics::counter!("distance_cache_misses").increment(1);
}

/// Discard a miss metric when metrics are not compiled.
#[cfg(not(feature = "metrics"))]
pub(super) const fn record_miss() {}

/// Record an LRU eviction.
#[cfg(feature = "metrics")]
pub(super) fn record_eviction() {
    metrics::counter!("distance_cache_evictions").increment(1);
}

/// Discard an eviction metric when metrics are not compiled.
#[cfg(not(feature = "metrics"))]
pub(super) const fn record_eviction() {}

/// Record cache lookup latency when a miss completes.
#[cfg(feature = "metrics")]
pub(super) fn record_lookup_latency(elapsed: Duration) {
    metrics::histogram!("distance_cache_lookup_latency_histogram").record(elapsed.as_secs_f64());
}

/// Discard a lookup-latency metric when metrics are not compiled.
#[cfg(not(feature = "metrics"))]
pub(super) const fn record_lookup_latency(_elapsed: Duration) {}
