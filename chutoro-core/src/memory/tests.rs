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

/// Asserts that a larger configuration produces a strictly larger estimate.
fn assert_estimate_grows(lower: u64, higher: u64, context: &str) {
    assert!(higher > lower, "expected {context}: {higher} > {lower}");
}

#[rstest]
#[case::m8_vs_m16(1_000, 8, 16)]
#[case::m16_vs_m24(1_000, 16, 24)]
fn estimate_grows_with_max_connections(
    #[case] point_count: usize,
    #[case] m_small: usize,
    #[case] m_large: usize,
) {
    assert_estimate_grows(
        estimate_peak_bytes(point_count, m_small),
        estimate_peak_bytes(point_count, m_large),
        &format!("M={m_large} versus M={m_small} at n={point_count}"),
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
    assert_estimate_grows(
        estimate_peak_bytes(n_small, max_connections),
        estimate_peak_bytes(n_large, max_connections),
        &format!("n={n_large} versus n={n_small} at M={max_connections}"),
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

#[cfg(feature = "cpu")]
#[rstest]
fn default_max_level_constant_matches_the_hnsw_defaults() {
    assert_eq!(
        DEFAULT_HNSW_MAX_LEVEL,
        crate::HnswParams::default().max_level(),
        "the memory model must track the level cap the sampler enforces"
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
    let harvest_width = m.saturating_mul(LEGACY_EF_PER_CONNECTION as u64);
    let edges = harvested_edge_count(n, harvest_width, default_layers_eighths(max_connections));
    let without_buffers = n
        .saturating_mul(2_u64.saturating_mul(m).saturating_mul(USIZE_BYTES))
        .saturating_add(n.saturating_mul(NODE_OVERHEAD_BYTES))
        .saturating_add(edges.saturating_mul(CANDIDATE_EDGE_BYTES))
        .saturating_add(n.saturating_mul(F32_BYTES))
        .saturating_add(edges.saturating_mul(CANDIDATE_EDGE_BYTES))
        .saturating_add(n.saturating_mul(MST_EDGE_BYTES))
        .saturating_mul(SAFETY_MULTIPLIER_NUMERATOR)
        .saturating_div(SAFETY_MULTIPLIER_DENOMINATOR);

    let buffer_contribution = edges
        .saturating_mul(VALIDATED_EDGE_SLOT_BYTES.saturating_add(MST_EDGE_BYTES))
        .saturating_mul(SAFETY_MULTIPLIER_NUMERATOR)
        .saturating_div(SAFETY_MULTIPLIER_DENOMINATOR);

    let estimate = estimate_peak_bytes_with_search_width(EstimateInputs {
        point_count,
        max_connections,
        effective_search_width: 0,
        distance_cache_capacity: 0,
        harvest_width: max_connections.saturating_mul(LEGACY_EF_PER_CONNECTION),
        layers_eighths: default_layers_eighths(max_connections),
        concurrent_searches: 1,
    });

    assert_eq!(
        estimate.saturating_sub(without_buffers),
        buffer_contribution,
        "n={point_count}, M={max_connections}: both prepare_edge_list buffers must \
         appear exactly once in the estimate"
    );
}

// -- multi-layer harvest model ------------------------------------------

/// Deterministic scalar data source used to probe real harvest sizes.
#[cfg(feature = "cpu")]
struct ScalarSource {
    values: Vec<f32>,
}

#[cfg(feature = "cpu")]
impl ScalarSource {
    /// Builds `len` pseudo-random non-negative values.
    ///
    /// Values are drawn as 16-bit integers and widened with `f32::from`, which
    /// is exact. Keeping the generator in integer arithmetic avoids the float
    /// restriction lints the workspace denies while still spreading the points
    /// far enough apart to exercise the graph.
    fn new(len: usize) -> Self {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let values = (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                f32::from(u16::try_from(state >> 48).unwrap_or(u16::MAX))
            })
            .collect();
        Self { values }
    }
}

#[cfg(feature = "cpu")]
impl crate::DataSource for ScalarSource {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn name(&self) -> &'static str {
        "scalar"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, crate::DataSourceError> {
        let a = self
            .values
            .get(left)
            .copied()
            .ok_or(crate::DataSourceError::OutOfBounds { index: left })?;
        let b = self
            .values
            .get(right)
            .copied()
            .ok_or(crate::DataSourceError::OutOfBounds { index: right })?;
        // `mul_add` expresses the subtraction without the float operators the
        // workspace restriction lints deny.
        Ok(a.mul_add(1.0, std::ops::Neg::neg(b)).abs())
    }
}

// The model describes the CPU HNSW build, so the probe needs the `cpu`
// feature even though the estimator itself is always compiled.
#[cfg(feature = "cpu")]
#[rstest]
#[case::default_parameters(16, 64)]
#[case::narrow_graph(4, 16)]
fn estimate_covers_multi_layer_harvests(
    #[case] max_connections: usize,
    #[case] ef_construction: usize,
) {
    const POINT_COUNT: usize = 500;
    let source = ScalarSource::new(POINT_COUNT);
    let params = crate::HnswParams::new(max_connections, ef_construction)
        .expect("test parameters must be valid");
    let (_, harvest) =
        crate::CpuHnsw::build_with_edges(&source, params).expect("the probe build must succeed");

    // `extract_candidate_edges` records candidates from every layer, so the
    // harvest is several times the single-layer `n × M` count.
    assert!(
        harvest.len() > POINT_COUNT.saturating_mul(max_connections),
        "the probe must exercise a multi-layer harvest, got {} edges",
        harvest.len()
    );

    let modelled = harvested_edge_count(
        POINT_COUNT as u64,
        ef_construction as u64,
        default_layers_eighths(max_connections),
    );
    assert!(
        (harvest.len() as u64) <= modelled,
        "harvest of {} edges must stay within the modelled {modelled}",
        harvest.len()
    );
}

// -- level expectation --------------------------------------------------

#[cfg(feature = "cpu")]
#[rstest]
fn default_parameters_match_the_geometric_layer_model() {
    let params = crate::HnswParams::default();

    assert_eq!(
        params.expected_layers_eighths(),
        default_layers_eighths(params.max_connections()),
        "the default sampler must reproduce the legacy geometric model"
    );
}

#[cfg(feature = "cpu")]
#[rstest]
fn a_capped_level_multiplier_raises_the_parameter_estimate() {
    let base = crate::HnswParams::new(16, 64).expect("test parameters must be valid");
    let capped = base.clone().with_level_multiplier(1.0e20);

    let base_bytes = estimate_peak_bytes_for_hnsw_params(1_000, &base);
    let capped_bytes = estimate_peak_bytes_for_hnsw_params(1_000, &capped);

    assert!(
        capped_bytes > base_bytes,
        "the capped layer count must raise the estimate, got {capped_bytes} <= {base_bytes}"
    );
}
