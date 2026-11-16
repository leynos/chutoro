//! Unit tests for memory accounting.
//!
//! Kept in a child module so the estimator source stays within the
//! repository's module length limit.

use super::*;
use rstest::rstest;

// -- estimate_peak_bytes: happy paths -----------------------------------

#[rstest]
#[case::small_m16(100, 16)]
#[case::medium_m8(1_000, 8)]
#[case::large_m16(1_000_000, 16)]
#[case::large_m24(1_000_000, 24)]
fn estimate_returns_positive_for_non_empty(
    #[case] point_count: usize,
    #[case] max_connections: usize,
) {
    let bytes = estimate_peak_bytes(point_count, max_connections);
    assert!(
        bytes > 0,
        "expected positive estimate for n={point_count}, M={max_connections}, got {bytes}"
    );
}

#[rstest]
#[case::m8_vs_m16(1_000, 8, 16)]
#[case::m16_vs_m24(1_000, 16, 24)]
fn estimate_grows_with_max_connections(
    #[case] point_count: usize,
    #[case] m_small: usize,
    #[case] m_large: usize,
) {
    let small = estimate_peak_bytes(point_count, m_small);
    let large = estimate_peak_bytes(point_count, m_large);
    assert!(
        large > small,
        "expected M={m_large} estimate ({large}) > M={m_small} estimate ({small})"
    );
}

#[cfg(feature = "cpu")]
#[rstest]
fn parameter_estimate_grows_with_effective_search_width() {
    let narrow = crate::HnswParams::new(4, 4).expect("parameters must be valid");
    let wide = crate::HnswParams::new(4, 64).expect("parameters must be valid");

    let narrow_bytes = estimate_peak_bytes_for_hnsw_params(100, &narrow);
    let wide_bytes = estimate_peak_bytes_for_hnsw_params(100, &wide);

    assert!(
        wide_bytes > narrow_bytes,
        "a wider effective search state must increase the memory estimate"
    );
}

#[cfg(feature = "cpu")]
#[rstest]
fn parameter_estimate_grows_with_distance_cache_capacity() {
    let default_params = crate::HnswParams::new(4, 16).expect("parameters must be valid");
    let cache_capacity = std::num::NonZeroUsize::new(
        crate::DistanceCacheConfig::DEFAULT_MAX_ENTRIES.saturating_mul(2),
    )
    .expect("doubled default cache capacity must be non-zero");
    let custom_cache_params = default_params
        .clone()
        .with_distance_cache_max_entries(cache_capacity);

    let default_bytes = estimate_peak_bytes_for_hnsw_params(100, &default_params);
    let custom_cache_bytes = estimate_peak_bytes_for_hnsw_params(100, &custom_cache_params);

    assert!(
        custom_cache_bytes > default_bytes,
        "a larger configured distance cache must increase the memory estimate"
    );
}

#[rstest]
#[case::hundred_vs_thousand(100, 1_000, 16)]
#[case::thousand_vs_million(1_000, 1_000_000, 16)]
fn estimate_grows_with_point_count(
    #[case] n_small: usize,
    #[case] n_large: usize,
    #[case] max_connections: usize,
) {
    let small = estimate_peak_bytes(n_small, max_connections);
    let large = estimate_peak_bytes(n_large, max_connections);
    assert!(
        large > small,
        "expected n={n_large} estimate ({large}) > n={n_small} estimate ({small})"
    );
}

// -- estimate_peak_bytes: edge cases ------------------------------------

#[rstest]
fn estimate_zero_points_returns_zero() {
    assert_eq!(estimate_peak_bytes(0, 16), 0);
}

#[rstest]
fn estimate_one_point_returns_positive_with_cache_base() {
    let bytes = estimate_peak_bytes(1, 16);
    assert!(bytes > 0, "single point should still have overhead");
    // The estimate includes the full distance cache base cost (~120 MiB),
    // so even a single point produces a sizeable estimate.
    assert!(
        bytes > 100_000_000,
        "expected cache base cost to dominate for n=1"
    );
}

#[rstest]
fn estimate_m_one_returns_valid() {
    let bytes = estimate_peak_bytes(1_000, 1);
    assert!(bytes > 0, "M=1 should still produce a positive estimate");
}

// -- estimate_peak_bytes: overflow protection ---------------------------

#[rstest]
fn estimate_huge_point_count_does_not_panic() {
    // Must not panic; saturating arithmetic should cap at u64::MAX.
    let bytes = estimate_peak_bytes(usize::MAX, 24);
    assert!(bytes > 0);
}

// -- prepare_edge_list buffers ------------------------------------------

// The slot constants describe MST types, which only exist with the `cpu`
// feature; the remaining memory tests stay available without it.
#[cfg(feature = "cpu")]
#[rstest]
fn buffer_slot_constants_cover_the_real_layouts() {
    // Compare in the `usize` domain so the constants keep their `u64`
    // declaration without a narrowing cast on 32-bit targets.
    let mst_edge_limit = usize::try_from(MST_EDGE_BYTES).unwrap_or(usize::MAX);
    let validated_slot_limit = usize::try_from(VALIDATED_EDGE_SLOT_BYTES).unwrap_or(usize::MAX);

    assert!(
        std::mem::size_of::<crate::MstEdge>() <= mst_edge_limit,
        "MST_EDGE_BYTES must not underestimate MstEdge"
    );
    assert!(
        std::mem::size_of::<Result<Option<crate::MstEdge>, crate::MstError>>()
            <= validated_slot_limit,
        "VALIDATED_EDGE_SLOT_BYTES must not underestimate the validation slot"
    );
}

#[rstest]
#[case::million_m16(1_000_000, 16)]
#[case::thousand_m8(1_000, 8)]
fn estimate_includes_the_prepare_edge_list_buffers(
    #[case] point_count: usize,
    #[case] max_connections: usize,
) {
    // The same shared terms the estimator sums before the safety
    // multiplier, minus the two `prepare_edge_list` buffers.
    let n = point_count as u64;
    let m = max_connections as u64;
    let without_buffers = n
        .saturating_mul(2_u64.saturating_mul(m).saturating_mul(USIZE_BYTES))
        .saturating_add(n.saturating_mul(NODE_OVERHEAD_BYTES))
        .saturating_add(n.saturating_mul(m).saturating_mul(CANDIDATE_EDGE_BYTES))
        .saturating_add(n.saturating_mul(F32_BYTES))
        .saturating_add(n.saturating_mul(m).saturating_mul(CANDIDATE_EDGE_BYTES))
        .saturating_add(n.saturating_mul(MST_EDGE_BYTES))
        .saturating_mul(SAFETY_MULTIPLIER_NUMERATOR)
        .saturating_div(SAFETY_MULTIPLIER_DENOMINATOR);

    let buffer_contribution = n
        .saturating_mul(m)
        .saturating_mul(VALIDATED_EDGE_SLOT_BYTES.saturating_add(MST_EDGE_BYTES))
        .saturating_mul(SAFETY_MULTIPLIER_NUMERATOR)
        .saturating_div(SAFETY_MULTIPLIER_DENOMINATOR);

    let estimate = estimate_peak_bytes_with_search_width(point_count, max_connections, 0, 0);

    assert_eq!(
        estimate.saturating_sub(without_buffers),
        buffer_contribution,
        "n={point_count}, M={max_connections}: both prepare_edge_list buffers must \
         appear exactly once in the estimate"
    );
}

// -- format_bytes -------------------------------------------------------

#[rstest]
#[case::zero(0, "0 B")]
#[case::small(512, "512 B")]
#[case::just_below_kib(1023, "1023 B")]
#[case::one_kib(1024, "1.0 KiB")]
#[case::rounds_half_to_even_down(1280, "1.2 KiB")]
#[case::one_and_half_kib(1536, "1.5 KiB")]
#[case::rounds_half_to_even_up(1792, "1.8 KiB")]
#[case::one_mib(1_048_576, "1.0 MiB")]
#[case::one_gib(1_073_741_824, "1.0 GiB")]
#[case::one_tib(1_099_511_627_776, "1.0 TiB")]
#[case::two_point_four_gib(2_576_980_378, "2.4 GiB")]
fn format_bytes_produces_expected_output(#[case] input: u64, #[case] expected: &str) {
    assert_eq!(format_bytes(input), expected);
}
