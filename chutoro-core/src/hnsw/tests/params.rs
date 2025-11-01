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
