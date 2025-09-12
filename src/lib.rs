//! Chutoro library.
//!
//! # Distance cache metrics
//!
//! The `DistanceCache` resolves neighbour ties deterministically: when
//! distances are equal it chooses the lower item id; if ids match it falls back
//! to the insertion sequence number. This rule ensures stable outputs under
//! fixed seeds.
//!
//! When the `metrics` feature is enabled the cache emits:
//!
//! - `distance_cache_hits` (counter)
//! - `distance_cache_misses` (counter)
//! - `distance_cache_evictions` (counter)
//! - `distance_cache_lookup_latency_histogram` (histogram, seconds)
//!
//! These metric names are stable for downstream crates.
/// Returns a greeting for the library.
#[must_use]
pub fn greet() -> &'static str {
    "Hello from Chutoro!"
}
