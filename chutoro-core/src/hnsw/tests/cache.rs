//! Tests for the concurrent distance cache supporting HNSW insertion.

use std::{error::Error, num::NonZeroUsize, thread, time::Duration};

use rstest::rstest;

use crate::{
    MetricDescriptor,
    hnsw::distance_cache::{DistanceCache, DistanceCacheConfig, LookupOutcome},
};

fn cache_with_capacity(capacity: usize) -> Result<DistanceCache, Box<dyn Error>> {
    let entries = NonZeroUsize::new(capacity).ok_or("capacity must be non-zero")?;
    Ok(DistanceCache::new(DistanceCacheConfig::new(entries)))
}

#[rstest]
fn caches_and_reuses_distances() {
    let cache = cache_with_capacity(4).expect("capacity must be non-zero");
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
    let cache = cache_with_capacity(2).expect("capacity must be non-zero");
    let metric = MetricDescriptor::new("lru");

    let LookupOutcome::Miss(miss_a) = cache.begin_lookup(&metric, 0, 1) else {
        panic!("initial cache lookup must miss");
    };
    cache
        .complete_miss(miss_a, 1.0)
        .expect("completing miss_a must succeed");

    let LookupOutcome::Miss(miss_b) = cache.begin_lookup(&metric, 0, 2) else {
        panic!("second cache lookup must miss");
    };
    cache
        .complete_miss(miss_b, 2.0)
        .expect("completing miss_b must succeed");

    let LookupOutcome::Miss(miss_c) = cache.begin_lookup(&metric, 0, 3) else {
        panic!("third cache lookup must miss");
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
        .with_ttl(Some(Duration::from_millis(20)));
    let cache = DistanceCache::new(config);
    let metric = MetricDescriptor::new("ttl");

    let LookupOutcome::Miss(miss) = cache.begin_lookup(&metric, 1, 2) else {
        panic!("initial cache lookup must miss");
    };
    cache
        .complete_miss(miss, 4.2)
        .expect("completing miss must succeed");

    thread::sleep(Duration::from_millis(100));

    match cache.begin_lookup(&metric, 1, 2) {
        LookupOutcome::Hit(_) => panic!("entry should expire after TTL"),
        LookupOutcome::Miss(_) => {}
    }
}

/// Ensures cache lookups hit regardless of the operand ordering, verifying key
/// normalization for symmetric distance metrics.
#[rstest]
fn normalizes_pair_order() {
    let cache = cache_with_capacity(2).expect("capacity must be non-zero");
    let metric = MetricDescriptor::new("sym");

    let LookupOutcome::Miss(miss) = cache.begin_lookup(&metric, 7, 3) else {
        panic!("initial cache lookup must miss");
    };
    cache
        .complete_miss(miss, 1.23)
        .expect("completing miss must succeed");

    match cache.begin_lookup(&metric, 3, 7) {
        LookupOutcome::Hit(value) => assert_eq!(value, 1.23),
        _ => panic!("normalized (a,b) must hit for (b,a)"),
    }
}

#[rstest]
fn rejects_non_finite_entries() {
    let cache = cache_with_capacity(1).expect("capacity must be non-zero");
    let metric = MetricDescriptor::new("nan");

    let LookupOutcome::Miss(miss) = cache.begin_lookup(&metric, 2, 3) else {
        panic!("initial cache lookup must miss");
    };
    let err = cache
        .complete_miss(miss, f32::NAN)
        .expect_err("NaN values must be rejected");
    assert!(matches!(
        err,
        crate::hnsw::HnswError::NonFiniteDistance { .. }
    ));
}

fn cache_config(max_entries: usize) -> Result<DistanceCacheConfig, Box<dyn Error>> {
    let entries = NonZeroUsize::new(max_entries).ok_or("non-zero")?;
    Ok(DistanceCacheConfig::new(entries))
}

#[rstest]
#[case(64, 64, None, true)]
#[case(64, 128, None, false)]
#[case(64, 64, Some(Duration::from_secs(1)), false)]
fn distance_cache_config_equality(
    #[case] left_entries: usize,
    #[case] right_entries: usize,
    #[case] right_ttl: Option<Duration>,
    #[case] expected_equal: bool,
) {
    let left_config = cache_config(left_entries).expect("non-zero");
    let mut right_config = cache_config(right_entries).expect("non-zero");
    if let Some(ttl) = right_ttl {
        right_config = right_config.with_ttl(Some(ttl));
    }

    if expected_equal {
        assert_eq!(left_config, right_config);
    } else {
        assert_ne!(left_config, right_config);
    }
}
