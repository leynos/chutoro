//! Peak-memory estimation for the CPU clustering pipeline.
//!
//! Holds the capacity model, the public entry points, and the shared
//! calculator that turns graph, search-state and cache capacities into a
//! conservative peak estimate.

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

/// Construction search breadth the legacy [`estimate_peak_bytes`] helper
/// assumes, expressed as a multiple of `M`.
///
/// That helper has no `ef_construction` parameter, so it models the documented
/// default pair (`M = 16`, `ef_construction = 64`). The sample interval sees
/// no better information, and four times `M` keeps the legacy estimate
/// meaningful across the `M` values in Table 4 of the design document.
const LEGACY_EF_PER_CONNECTION: usize = 4;

/// Highest graph level the default HNSW level sampler can reach.
///
/// Mirrors `HnswParams::default().max_level()`. The memory module compiles
/// without the `cpu` feature, so it cannot import the parameter type; a
/// `cpu`-gated unit test keeps the two values in step.
const DEFAULT_HNSW_MAX_LEVEL: usize = 12;

/// Fixed-point scale used to keep the expected layer count in integer
/// arithmetic within [`estimate_peak_bytes_with_search_width`].
const LAYER_FRACTION_SCALE: u64 = 8;

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
/// - Candidate edges harvested during HNSW build (`≈ n × ef × layers`).
/// - Core-distance vector (`n × sizeof(f32)`).
/// - Mutual-reachability edge rewrite (same count as candidate edges).
/// - The ordered validation buffer built by `prepare_edge_list`
///   (one `Result<Option<MstEdge>, MstError>` slot per harvested edge).
/// - The prepared edge list retained for sorting and deduplication
///   (one `MstEdge` per harvested edge).
/// - MST forest edges (`n` edges, rounding up from `n − 1`).
///
/// The harvested edge count is the multi-layer count, not `n × M`.
/// `extract_candidate_edges` records up to `ef` search results from every
/// layer a node occupies, duplicates included, so a healthy build harvests
/// roughly `n × ef` edges. `harvested_edge_count` models that with the level
/// sampler's geometric distribution; see its documentation for the exact
/// assumption.
///
/// This helper has no parameter object at all, so it models the documented
/// default construction ratio of four times `M` and the default level
/// multiplier of `1 / ln(M)`. Prefer
/// `estimate_peak_bytes_for_hnsw_params` when the construction breadth, level
/// cap or level multiplier are known.
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
    estimate_peak_bytes_with_search_width(EstimateInputs {
        point_count,
        max_connections,
        effective_search_width: 0,
        distance_cache_capacity: DEFAULT_CACHE_MAX_ENTRIES,
        harvest_width: max_connections.saturating_mul(LEGACY_EF_PER_CONNECTION),
        layers_eighths: default_layers_eighths(max_connections),
        concurrent_searches: 1,
    })
}

/// Returns the guarded peak estimate for concrete CPU HNSW parameters.
///
/// This extends [`estimate_peak_bytes`] with the temporary search-state
/// allocation and distance-cache capacity configured for the CPU HNSW index.
/// It also derives the expected layer count from the configured level
/// multiplier, so a multiplier that drives every node to `max_level` is
/// counted at the full `max_level + 1` layers rather than at the default
/// geometric mean.
///
/// It assumes one live search state; use
/// [`estimate_peak_bytes_for_hnsw_params_with_concurrency`](crate::estimate_peak_bytes_for_hnsw_params_with_concurrency) when several
/// searches can run at once, as the parallel batch core-distance stage does.
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
    let effective_search_width = hnsw_params.effective_ef_construction(point_count);
    estimate_peak_bytes_with_search_width(EstimateInputs {
        point_count,
        max_connections: hnsw_params.max_connections(),
        effective_search_width,
        distance_cache_capacity: hnsw_params.distance_cache_config().max_entries().get(),
        // Layer searches retain up to `ef` candidates each, so the harvest
        // width matches the bounded construction breadth.
        harvest_width: effective_search_width,
        // The parameters own the sampler, so their expectation follows an
        // overridden level multiplier, including one that reaches the cap.
        layers_eighths: hnsw_params.expected_layers_eighths(),
        concurrent_searches: 1,
    })
}

/// Returns the expected number of HNSW layers a node occupies, in eighths,
/// for the default level sampler.
///
/// The default level multiplier is `1 / ln(M)`, so a node continues to the
/// next layer with probability `1 / M` and the expected count is
/// `1 + 1/M + 1/M² + …`, which converges to `M / (M − 1)`; sampling is also
/// capped at [`DEFAULT_HNSW_MAX_LEVEL`]. With `M = 1` every draw continues, so
/// each node reaches the cap. The result is rounded up in eighths (at most
/// 12.5% above the exact expectation) so the estimator can stay in integer
/// arithmetic.
///
/// This models the legacy entry point, which has no parameter object. The
/// parameter-aware estimate asks [`crate::HnswParams`] for the expectation of
/// its configured sampler, so an overridden level multiplier is honoured.
const fn default_layers_eighths(max_connections: usize) -> u64 {
    let layer_cap = (DEFAULT_HNSW_MAX_LEVEL as u64)
        .saturating_add(1)
        .saturating_mul(LAYER_FRACTION_SCALE);
    if max_connections <= 1 {
        return layer_cap;
    }
    let m = max_connections as u64;
    // ceil(8 × M / (M − 1)), expressed with integer arithmetic.
    let geometric = LAYER_FRACTION_SCALE
        .saturating_mul(m)
        .saturating_add(m.saturating_sub(2))
        .div_euclid(m - 1);
    if geometric < layer_cap {
        geometric
    } else {
        layer_cap
    }
}

/// Returns the expected number of candidate edges an HNSW build harvests.
///
/// `extract_candidate_edges` records up to `harvest_width` search results from
/// every layer a node occupies, and the same neighbour may appear in several
/// layers, so the harvest is `point_count × harvest_width × layers` rather
/// than one fan-out per node, where `layers_eighths` is the expected layer
/// count in eighths. The model is an expectation: a single node can occupy up
/// to `max_level + 1` layers, which is the strict worst case, while the
/// sampler's distribution concentrates the total near its mean at the dataset
/// sizes the guard serves. The 1.5× safety multiplier in the caller leaves
/// headroom above that mean.
const fn harvested_edge_count(point_count: u64, harvest_width: u64, layers_eighths: u64) -> u64 {
    point_count
        .saturating_mul(harvest_width)
        .saturating_mul(layers_eighths)
        .div_euclid(LAYER_FRACTION_SCALE)
}

/// Inputs to the shared peak-memory estimate.
///
/// Grouping them keeps the estimator signature readable and lets the two
/// public entry points differ only in the fields they supply.
#[derive(Clone, Copy)]
pub(super) struct EstimateInputs {
    /// Number of points in the batch.
    pub(super) point_count: usize,
    /// HNSW `M` parameter.
    pub(super) max_connections: usize,
    /// Bound on the construction search breadth, used for search state.
    pub(super) effective_search_width: usize,
    /// Configured distance-cache capacity, in entries.
    pub(super) distance_cache_capacity: usize,
    /// Candidates retained by each layer search.
    pub(super) harvest_width: usize,
    /// Expected layers a node occupies, in eighths.
    pub(super) layers_eighths: u64,
    /// Search states that can be live at the same time; at least one.
    pub(super) concurrent_searches: usize,
}

/// Calculates the shared estimate from graph, search-state, and cache capacities.
pub(super) const fn estimate_peak_bytes_with_search_width(inputs: EstimateInputs) -> u64 {
    let EstimateInputs {
        point_count,
        max_connections,
        effective_search_width,
        distance_cache_capacity,
        harvest_width,
        layers_eighths,
        concurrent_searches,
    } = inputs;
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

    // Candidate edges harvested across every layer the inserted nodes occupy.
    let harvested_edges = harvested_edge_count(n, harvest_width as u64, layers_eighths);
    let candidate_edges = harvested_edges.saturating_mul(CANDIDATE_EDGE_BYTES);

    // Core-distance vector: one f32 per point.
    let core_distances = n.saturating_mul(F32_BYTES);

    // Mutual-reachability rewrite: same count as candidate edges.
    let mutual_edges = candidate_edges;

    // Ordered validation buffer: one `Result<Option<MstEdge>, MstError>` slot
    // per harvested edge, sized before the parallel map fills it in place.
    let validated_edge_slots = harvested_edges.saturating_mul(VALIDATED_EDGE_SLOT_BYTES);

    // Prepared edge list: the surviving `MstEdge` values are copied out of the
    // validation buffer before it is dropped, so both exist at the same time.
    let prepared_edges = harvested_edges.saturating_mul(MST_EDGE_BYTES);

    // MST forest: up to n edges (n − 1 for a connected graph, rounded up).
    let mst_forest = n.saturating_mul(MST_EDGE_BYTES);

    // CPU construction creates search queues sized by the effective `ef`, and
    // the parallel core-distance stage keeps one such queue set per worker.
    let concurrent_states = if concurrent_searches == 0 {
        1
    } else {
        concurrent_searches
    };
    let search_state = search_width
        .saturating_mul(SEARCH_STATE_BYTES_PER_WIDTH)
        .saturating_mul(concurrent_states as u64);

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

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
