//! Pre-flight memory estimation for the CPU clustering pipeline.
//!
//! Provides a conservative estimate of peak memory consumption so callers can
//! reject oversized datasets before any allocation occurs.  The estimate is
//! intentionally pessimistic — it uses a safety multiplier to account for heap
//! fragmentation, Rayon thread-local buffers, and temporary allocations that
//! are difficult to predict statically.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Safety multiplier applied to the raw estimate to cover heap fragmentation,
/// Rayon thread-local buffers, and transient allocations.  1.5× is chosen as
/// a balance between avoiding false positives and catching genuine OOM risks.
const SAFETY_MULTIPLIER_NUMERATOR: u64 = 3;
/// Denominator paired with the safety multiplier numerator.
const SAFETY_MULTIPLIER_DENOMINATOR: u64 = 2;

/// Default maximum distance-cache entries used by [`estimate_peak_bytes`].
///
/// This mirrors `DistanceCacheConfig::DEFAULT_MAX_ENTRIES` while allowing the
/// legacy estimator to compile without the `cpu` feature gate. Parameter-aware
/// estimates use the capacity configured on [`crate::HnswParams`] instead.
const DEFAULT_CACHE_MAX_ENTRIES: usize = 1_048_576;

/// Estimated overhead per node in the HNSW graph: `Option<Node>`, `Vec`
/// headers for the per-level neighbour lists, sequence counter, and alignment
/// padding.  Derived from the layout of `hnsw::graph::Node` on 64-bit Linux.
const NODE_OVERHEAD_BYTES: u64 = 80;

/// Size of a single `CandidateEdge` (`source: usize`, `target: usize`,
/// `distance: f32`, `sequence: u64`) including padding on 64-bit platforms.
const CANDIDATE_EDGE_BYTES: u64 = 32;

/// Size of a single `MstEdge` (identical layout to `CandidateEdge`).
const MST_EDGE_BYTES: u64 = 32;

/// Estimated per-entry overhead for the distance cache, accounting for the
/// `DashMap` slot, the `LruCache` bookkeeping, and the stored key/value.
const CACHE_ENTRY_BYTES: u64 = 80;

/// Size of an `f32` — used for the core-distances vector.
const F32_BYTES: u64 = 4;

/// Conservative allocation budget for one `SearchState` width unit. It covers
/// two binary heaps and two hash sets, including their expected spare capacity.
const SEARCH_STATE_BYTES_PER_WIDTH: u64 = 256;
/// Size of a `usize` — derived at compile time so the estimate adapts to the
/// target platform (8 bytes on 64-bit, 4 bytes on 32-bit).
const USIZE_BYTES: u64 = std::mem::size_of::<usize>() as u64;

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

/// Returns a conservative estimate of peak memory (in bytes) that the CPU
/// pipeline will require for `point_count` items with the given HNSW
/// `max_connections` parameter (`M`).
///
/// The estimate covers:
///
/// - HNSW level-0 adjacency lists (`2 × M` neighbours per node).
/// - Per-node struct overhead (Vec headers, sequence counter, alignment).
/// - Distance cache at `DistanceCacheConfig::DEFAULT_MAX_ENTRIES` capacity.
/// - Candidate edges harvested during HNSW build (`≈ n × M`).
/// - Core-distance vector (`n × sizeof(f32)`).
/// - Mutual-reachability edge rewrite (same count as candidate edges).
/// - MST forest edges (`n` edges, rounding up from `n − 1`).
///
/// A 1.5× safety multiplier is applied to the raw total to account for heap
/// fragmentation, Rayon thread-local buffers, and transient allocations.
///
/// # Examples
///
/// ```
/// use chutoro_core::estimate_peak_bytes;
///
/// let bytes = estimate_peak_bytes(1_000, 16);
/// assert!(bytes > 0, "estimate must be positive for non-empty datasets");
///
/// let zero = estimate_peak_bytes(0, 16);
/// assert_eq!(zero, 0, "empty dataset requires no memory");
/// ```
#[must_use]
pub const fn estimate_peak_bytes(point_count: usize, max_connections: usize) -> u64 {
    estimate_peak_bytes_with_search_width(
        point_count,
        max_connections,
        0,
        DEFAULT_CACHE_MAX_ENTRIES,
    )
}

/// Returns the guarded peak estimate for concrete CPU HNSW parameters.
///
/// This extends [`estimate_peak_bytes`] with the temporary search-state
/// allocation and distance-cache capacity configured for the CPU HNSW index.
///
/// # Examples
///
/// ```
/// use chutoro_core::{HnswParams, estimate_peak_bytes_for_hnsw_params};
///
/// let params = HnswParams::new(16, 64).expect("parameters must be valid");
/// let bytes = estimate_peak_bytes_for_hnsw_params(1_000, &params);
/// assert!(bytes > 0, "a non-empty CPU run requires memory");
/// ```
#[cfg(feature = "cpu")]
#[must_use]
pub fn estimate_peak_bytes_for_hnsw_params(
    point_count: usize,
    hnsw_params: &crate::HnswParams,
) -> u64 {
    estimate_peak_bytes_with_search_width(
        point_count,
        hnsw_params.max_connections(),
        hnsw_params.effective_ef_construction(point_count),
        hnsw_params.distance_cache_config().max_entries().get(),
    )
}

const fn estimate_peak_bytes_with_search_width(
    point_count: usize,
    max_connections: usize,
    search_width: usize,
    distance_cache_capacity: usize,
) -> u64 {
    if point_count == 0 {
        return 0;
    }

    let n = point_count as u64;
    let m = max_connections as u64;
    let search_width = search_width as u64;

    // HNSW level-0 adjacency: each node keeps up to 2*M neighbour IDs.
    let hnsw_adjacency = n.saturating_mul(2_u64.saturating_mul(m).saturating_mul(USIZE_BYTES));

    // Per-node struct overhead (Option<Node>, Vec headers, sequence, etc.).
    let hnsw_nodes = n.saturating_mul(NODE_OVERHEAD_BYTES);

    // Pairwise lookups can fill the configured cache capacity even for small
    // batches, so account for every configured entry rather than point count.
    let distance_cache = (distance_cache_capacity as u64).saturating_mul(CACHE_ENTRY_BYTES);

    // Candidate edges: approximately n * M edges from the HNSW build.
    let candidate_edges = n.saturating_mul(m).saturating_mul(CANDIDATE_EDGE_BYTES);

    // Core-distance vector: one f32 per point.
    let core_distances = n.saturating_mul(F32_BYTES);

    // Mutual-reachability rewrite: same count as candidate edges.
    let mutual_edges = candidate_edges;

    // MST forest: up to n edges (n − 1 for a connected graph, rounded up).
    let mst_forest = n.saturating_mul(MST_EDGE_BYTES);

    // CPU construction creates search queues sized by the effective `ef`.
    let search_state = search_width.saturating_mul(SEARCH_STATE_BYTES_PER_WIDTH);

    let subtotal = hnsw_adjacency
        .saturating_add(hnsw_nodes)
        .saturating_add(distance_cache)
        .saturating_add(candidate_edges)
        .saturating_add(core_distances)
        .saturating_add(mutual_edges)
        .saturating_add(mst_forest)
        .saturating_add(search_state);

    // Apply safety multiplier (3/2 = 1.5×) using integer arithmetic.
    subtotal
        .saturating_mul(SAFETY_MULTIPLIER_NUMERATOR)
        .saturating_div(SAFETY_MULTIPLIER_DENOMINATOR)
}

/// Number of bytes in one kibibyte.
const KIB: u64 = 1024;
/// Number of bytes in one mebibyte.
const MIB: u64 = 1024 * KIB;
/// Number of bytes in one gibibyte.
const GIB: u64 = 1024 * MIB;
/// Number of bytes in one tebibyte.
const TIB: u64 = 1024 * GIB;

/// Selects the appropriate binary unit and divisor for a byte count.
const fn binary_unit(bytes: u64) -> (&'static str, u64) {
    if bytes >= TIB {
        ("TiB", TIB)
    } else if bytes >= GIB {
        ("GiB", GIB)
    } else if bytes >= MIB {
        ("MiB", MIB)
    } else {
        ("KiB", KIB)
    }
}

/// Formats a byte count as a human-readable string using binary units.
///
/// Returns values like `"0 B"`, `"1.0 KiB"`, `"2.4 GiB"`.  The result uses
/// one decimal place for values ≥ 1 KiB.
///
/// # Examples
///
/// ```
/// use chutoro_core::format_bytes;
///
/// assert_eq!(format_bytes(0), "0 B");
/// assert_eq!(format_bytes(1023), "1023 B");
/// assert_eq!(format_bytes(1024), "1.0 KiB");
/// assert_eq!(format_bytes(1_073_741_824), "1.0 GiB");
/// ```
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    if bytes < KIB {
        return format!("{bytes} B");
    }
    let (label, divisor) = binary_unit(bytes);
    let whole = bytes.div_euclid(divisor);
    let scaled_remainder = bytes.rem_euclid(divisor).saturating_mul(10);
    let tenths = scaled_remainder.div_euclid(divisor);
    let remaining_fraction = scaled_remainder.rem_euclid(divisor);
    let half_divisor = divisor.div_euclid(2);
    let should_round_up = remaining_fraction > half_divisor
        || (remaining_fraction == half_divisor && tenths.rem_euclid(2) == 1);

    if should_round_up && tenths == 9 {
        format!("{}.0 {label}", whole.saturating_add(1))
    } else {
        let displayed_tenths = tenths + u64::from(should_round_up);
        format!("{whole}.{displayed_tenths} {label}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Unit tests for memory accounting.

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
}
