//! Tests for the concurrent distance cache supporting HNSW insertion.

use std::{num::NonZeroUsize, thread, time::Duration};

use rstest::rstest;

use crate::{
    MetricDescriptor,
    hnsw::distance_cache::{DistanceCache, DistanceCacheConfig, LookupOutcome},
};

fn cache_with_capacity(capacity: usize) -> DistanceCache {
    let config =
        DistanceCacheConfig::new(NonZeroUsize::new(capacity).expect("capacity must be non-zero"));
    DistanceCache::new(config)
}

#[rstest]
fn caches_and_reuses_distances() {
    let cache = cache_with_capacity(4);
    let metric = MetricDescriptor::new("test-metric");

    let miss = match cache.begin_lookup(&metric, 0, 1) {
        LookupOutcome::Hit(_) => panic!("cache should be empty on first lookup"),
        LookupOutcome::Miss(miss) => miss,
    };
    cache
        .complete_miss(miss, 0.5)
        .expect("completing miss must succeed");

    match cache.begin_lookup(&metric, 0, 1) {
        LookupOutcome::Hit(value) => assert_eq!(value, 0.5),
        LookupOutcome::Miss(_) => panic!("value should have been cached"),
    }
}

#[rstest]
fn lru_eviction_discards_oldest_entry() {
    let cache = cache_with_capacity(2);
    let metric = MetricDescriptor::new("lru");

    let miss_a = match cache.begin_lookup(&metric, 0, 1) {
        LookupOutcome::Miss(miss) => miss,
        _ => unreachable!(),
    };
    cache
        .complete_miss(miss_a, 1.0)
        .expect("completing miss_a must succeed");

    let miss_b = match cache.begin_lookup(&metric, 0, 2) {
        LookupOutcome::Miss(miss) => miss,
        _ => unreachable!(),
    };
    cache
        .complete_miss(miss_b, 2.0)
        .expect("completing miss_b must succeed");

    let miss_c = match cache.begin_lookup(&metric, 0, 3) {
        LookupOutcome::Miss(miss) => miss,
        _ => unreachable!(),
    };
    cache
        .complete_miss(miss_c, 3.0)
        .expect("completing miss_c must succeed");

    match cache.begin_lookup(&metric, 0, 1) {
        LookupOutcome::Hit(_) => panic!("oldest entry should have been evicted"),
        LookupOutcome::Miss(_) => {}
    }
    match cache.begin_lookup(&metric, 0, 2) {
        LookupOutcome::Hit(value) => assert_eq!(value, 2.0),
        LookupOutcome::Miss(_) => panic!("recent entry must be retained"),
    }
    match cache.begin_lookup(&metric, 0, 3) {
        LookupOutcome::Hit(value) => assert_eq!(value, 3.0),
        LookupOutcome::Miss(_) => panic!("new entry must be present"),
    }
}

#[rstest]
fn ttl_expiry_forces_refresh() {
    let config = DistanceCacheConfig::new(NonZeroUsize::new(2).expect("capacity"))
        .with_ttl(Some(Duration::from_millis(1)));
    let cache = DistanceCache::new(config);
    let metric = MetricDescriptor::new("ttl");

    let miss = match cache.begin_lookup(&metric, 1, 2) {
        LookupOutcome::Miss(miss) => miss,
        _ => unreachable!(),
    };
    cache
        .complete_miss(miss, 4.2)
        .expect("completing miss must succeed");

    thread::sleep(Duration::from_millis(5));

    match cache.begin_lookup(&metric, 1, 2) {
        LookupOutcome::Hit(_) => panic!("entry should expire after TTL"),
        LookupOutcome::Miss(_) => {}
    }
}

#[rstest]
fn rejects_non_finite_entries() {
    let cache = cache_with_capacity(1);
    let metric = MetricDescriptor::new("nan");

    let miss = match cache.begin_lookup(&metric, 2, 3) {
        LookupOutcome::Miss(miss) => miss,
        _ => unreachable!(),
    };
    let err = cache
        .complete_miss(miss, f32::NAN)
        .expect_err("NaN values must be rejected");
    assert!(matches!(
        err,
        crate::hnsw::HnswError::NonFiniteDistance { .. }
    ));
}
