//! Compile-pass contract for representative public `const fn` APIs.
//!
//! This fixture deliberately avoids [`ChutoroBuilder::new`], which is not a
//! `const fn`. It instead verifies public APIs that are explicitly available
//! to callers in constant declarations.

use std::num::NonZeroUsize;

use chutoro_core::{
    ClusterId, DistanceCacheConfig, HierarchyConfig, estimate_peak_bytes,
};

const PEAK_BYTES: u64 = estimate_peak_bytes(1_000, 16);
const CLUSTER_ID: ClusterId = ClusterId::new(42);
const CLUSTER_VALUE: u64 = CLUSTER_ID.get();
const HIERARCHY_CONFIG: HierarchyConfig = HierarchyConfig::new(NonZeroUsize::MIN);
const HIERARCHY_MIN_CLUSTER_SIZE: NonZeroUsize = HIERARCHY_CONFIG.min_cluster_size();
const CACHE_CONFIG: DistanceCacheConfig =
    DistanceCacheConfig::new(NonZeroUsize::MIN).with_max_entries(NonZeroUsize::MIN);
const CACHE_MAX_ENTRIES: NonZeroUsize = CACHE_CONFIG.max_entries();

fn main() {
    let _ = (
        PEAK_BYTES,
        CLUSTER_VALUE,
        HIERARCHY_MIN_CLUSTER_SIZE,
        CACHE_MAX_ENTRIES,
    );
}
