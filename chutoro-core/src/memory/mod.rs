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

/// Estimated size of one slot in the ordered validation buffer built by
/// `prepare_edge_list`: a `Result<Option<MstEdge>, MstError>`.
///
/// The `Result` discriminant, the `Option` discriminant, and the largest
/// `MstError` variant (`InvariantViolation`, which carries a `&'static str`
/// and two `usize` values) together dominate the payload, so this is larger
/// than [`MST_EDGE_BYTES`]. A unit test pins the constant against the real
/// layout so it cannot silently underestimate the buffer.
const VALIDATED_EDGE_SLOT_BYTES: u64 = 48;

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
/// - The ordered validation buffer built by `prepare_edge_list`
///   (`≈ n × M` slots of `Result<Option<MstEdge>, MstError>`).
/// - The prepared edge list retained for sorting and deduplication
///   (`≈ n × M` `MstEdge` values).
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

/// Calculates the shared estimate from graph, search-state, and cache capacities.
const fn estimate_peak_bytes_with_search_width(
    point_count: usize,
    max_connections: usize,
    effective_search_width: usize,
    distance_cache_capacity: usize,
) -> u64 {
    if point_count == 0 {
        return 0;
    }

    let n = point_count as u64;
    let m = max_connections as u64;
    let search_width = effective_search_width as u64;

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

    // Ordered validation buffer: one `Result<Option<MstEdge>, MstError>` slot
    // per candidate edge, sized before the parallel map fills it in place.
    let validated_edge_slots = n
        .saturating_mul(m)
        .saturating_mul(VALIDATED_EDGE_SLOT_BYTES);

    // Prepared edge list: the surviving `MstEdge` values are copied out of the
    // validation buffer before it is dropped, so both exist at the same time.
    let prepared_edges = n.saturating_mul(m).saturating_mul(MST_EDGE_BYTES);

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
        .saturating_add(validated_edge_slots)
        .saturating_add(prepared_edges)
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
mod tests;
