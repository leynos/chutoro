//! Parameter validation tests for HNSW configuration structures.

use std::{num::NonZeroUsize, time::Duration};

use crate::hnsw::HnswParams;

#[test]
fn accepts_equal_search_and_connection_width() {
    let params = HnswParams::new(8, 8).expect("equal widths must be valid");
    assert_eq!(params.max_connections(), 8);
    assert_eq!(params.ef_construction(), 8);
}

#[test]
fn effective_ef_construction_preserves_minimum_search_width() {
    let params = HnswParams::new(4, 64).expect("parameters must be valid");

    assert_eq!(params.effective_ef_construction(3), 4);
    assert_eq!(params.effective_ef_construction(12), 12);
    assert_eq!(params.effective_ef_construction(128), 64);
    assert_eq!(
        params.clone().bounded_for_point_count(12).ef_construction(),
        12
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn accepts_connection_widths_above_u32_range() {
    let width = usize::try_from(u64::from(u32::MAX) + 1).expect("64-bit usize must fit width");

    let params = HnswParams::new(width, width).expect("usize-sized widths must be valid");

    assert_eq!(params.max_connections(), width);
    assert_eq!(params.ef_construction(), width);
}

#[test]
fn preserves_distance_cache_ttl_when_overriding_capacity() {
    let ttl = Some(Duration::from_secs(5));
    let params = HnswParams::new(8, 16)
        .expect("parameters must be valid")
        .with_distance_cache_ttl(ttl)
        .with_distance_cache_max_entries(
            NonZeroUsize::new(32).expect("max entries must be non-zero"),
        );

    let config = params.distance_cache_config();
    assert_eq!(config.ttl(), ttl, "TTL must survive capacity overrides");
    assert_eq!(config.max_entries().get(), 32);
}
