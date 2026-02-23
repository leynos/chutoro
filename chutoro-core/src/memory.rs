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
const SAFETY_MULTIPLIER_DENOMINATOR: u64 = 2;

/// Default maximum distance cache entries.  Mirrors the value in
/// `DistanceCacheConfig::DEFAULT_MAX_ENTRIES` but is duplicated here so the
/// estimation module compiles without the `cpu` feature gate.
const DEFAULT_CACHE_MAX_ENTRIES: u64 = 1_048_576;

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

/// Size of a `usize` on 64-bit platforms — used for neighbour-list entries.
const USIZE_BYTES: u64 = 8;

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
/// - Distance cache (bounded by the default maximum of 1,048,576 entries).
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
pub fn estimate_peak_bytes(point_count: usize, max_connections: usize) -> u64 {
    if point_count == 0 {
        return 0;
    }

    let n = point_count as u64;
    let m = max_connections as u64;

    // HNSW level-0 adjacency: each node keeps up to 2*M neighbour IDs.
    let hnsw_adjacency = n.saturating_mul(2_u64.saturating_mul(m).saturating_mul(USIZE_BYTES));

    // Per-node struct overhead (Option<Node>, Vec headers, sequence, etc.).
    let hnsw_nodes = n.saturating_mul(NODE_OVERHEAD_BYTES);

    // Distance cache — bounded by DEFAULT_CACHE_MAX_ENTRIES.
    let cache_entries = n.min(DEFAULT_CACHE_MAX_ENTRIES);
    let distance_cache = cache_entries.saturating_mul(CACHE_ENTRY_BYTES);

    // Candidate edges: approximately n * M edges from the HNSW build.
    let candidate_edges = n.saturating_mul(m).saturating_mul(CANDIDATE_EDGE_BYTES);

    // Core-distance vector: one f32 per point.
    let core_distances = n.saturating_mul(F32_BYTES);

    // Mutual-reachability rewrite: same count as candidate edges.
    let mutual_edges = candidate_edges;

    // MST forest: up to n edges (n − 1 for a connected graph, rounded up).
    let mst_forest = n.saturating_mul(MST_EDGE_BYTES);

    let subtotal = hnsw_adjacency
        .saturating_add(hnsw_nodes)
        .saturating_add(distance_cache)
        .saturating_add(candidate_edges)
        .saturating_add(core_distances)
        .saturating_add(mutual_edges)
        .saturating_add(mst_forest);

    // Apply safety multiplier (3/2 = 1.5×) using integer arithmetic.
    subtotal
        .saturating_mul(SAFETY_MULTIPLIER_NUMERATOR)
        .saturating_div(SAFETY_MULTIPLIER_DENOMINATOR)
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

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
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    const TIB: u64 = 1024 * GIB;

    if bytes >= TIB {
        format!("{:.1} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
    fn estimate_one_point_returns_small_positive() {
        let bytes = estimate_peak_bytes(1, 16);
        assert!(bytes > 0, "single point should still have overhead");
        assert!(bytes < 1_000_000, "single point should be well under 1 MiB");
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
    #[case::one_and_half_kib(1536, "1.5 KiB")]
    #[case::one_mib(1_048_576, "1.0 MiB")]
    #[case::one_gib(1_073_741_824, "1.0 GiB")]
    #[case::one_tib(1_099_511_627_776, "1.0 TiB")]
    #[case::two_point_four_gib(2_576_980_378, "2.4 GiB")]
    fn format_bytes_produces_expected_output(#[case] input: u64, #[case] expected: &str) {
        assert_eq!(format_bytes(input), expected);
    }
}
