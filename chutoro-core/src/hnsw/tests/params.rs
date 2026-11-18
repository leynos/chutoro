//! Parameter validation tests for HNSW configuration structures.

use std::{num::NonZeroUsize, time::Duration};

use rstest::rstest;

use crate::hnsw::HnswParams;

/// Fixed-point scale used by `HnswParams::expected_layers_eighths`.
const EIGHTHS_PER_LAYER: u64 = 8;

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

#[rstest]
fn a_large_level_multiplier_uses_the_level_cap() {
    // A very large multiplier keeps every draw below the sampler's stop
    // threshold, so every node occupies all `max_level + 1` layers.
    let params = HnswParams::new(16, 64)
        .expect("test parameters must be valid")
        .with_level_multiplier(1.0e20);
    let cap = u64::try_from(params.max_level())
        .expect("the level cap must fit u64")
        .saturating_add(1)
        .saturating_mul(EIGHTHS_PER_LAYER);

    assert_eq!(
        params.expected_layers_eighths(),
        cap,
        "a cap-reaching multiplier must model every permitted layer"
    );
}

#[rstest]
fn a_tiny_level_multiplier_keeps_nodes_at_level_zero() {
    // A multiplier this small underflows the continuation probability to
    // zero, so every node stops at level zero.
    let params = HnswParams::default().with_level_multiplier(f64::MIN_POSITIVE);

    assert_eq!(
        params.expected_layers_eighths(),
        EIGHTHS_PER_LAYER,
        "a never-continuing sampler models exactly one layer"
    );
}

#[rstest]
fn a_multiplier_below_the_sampler_minimum_keeps_nodes_at_level_zero() {
    // `should_stop` accepts the minimum clamped draw at this multiplier, so no
    // node can reach level one even though `exp(-1 / multiplier)` is not zero.
    let params = HnswParams::default().with_level_multiplier(0.035);

    assert_eq!(
        params.expected_layers_eighths(),
        EIGHTHS_PER_LAYER,
        "a sampler that always stops models exactly one layer"
    );
}

#[rstest]
fn a_capped_sampler_uses_the_finite_series() {
    // `q = exp(-0.1)` over ten permitted layers gives
    // `(1 - q^10) / (1 - q) = (1 - e^-1) / (1 - e^-0.1)`, about 6.6425
    // layers, which rounds up to 54 eighths. The infinite series would report
    // the full ten-layer cap instead.
    let params = HnswParams::new(16, 64)
        .expect("test parameters must be valid")
        .with_max_level(9)
        .with_level_multiplier(10.0);

    assert_eq!(
        params.expected_layers_eighths(),
        54,
        "a binding cap must use the finite expectation, not the cap itself"
    );
}

#[rstest]
fn a_single_level_index_reports_one_layer() {
    let params = HnswParams::default().with_max_level(0);

    assert_eq!(
        params.expected_layers_eighths(),
        EIGHTHS_PER_LAYER,
        "level zero is the only permitted layer"
    );
}
