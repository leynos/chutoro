//! Batch-aware peak-memory estimation.
//!
//! The batch core-distance stage searches at a width derived from the cluster
//! size and the construction breadth, and it runs one search per Rayon worker.
//! Both facts shape the pre-flight guard, so they live here rather than beside
//! the single-search estimator.

use std::num::NonZeroUsize;

use super::estimate::{EstimateInputs, estimate_peak_bytes_with_search_width};

/// Returns the search width the batch core-distance stage uses per point.
///
/// The stage looks for each point's `min_cluster_size + 1` nearest
/// neighbours, so a large cluster size raises the search width above the
/// construction breadth. The width is capped at the number of points and is
/// never zero.
///
/// The pipeline and the memory guard share this rule so a change to the batch
/// width cannot silently desynchronise the two.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use chutoro_core::{HnswParams, batch_search_width};
///
/// let params = HnswParams::new(4, 16).expect("parameters must be valid");
/// let cluster = NonZeroUsize::new(64).expect("sixty-four is non-zero");
/// assert_eq!(batch_search_width(1_000, cluster, &params).get(), 65);
/// assert_eq!(batch_search_width(8, cluster, &params).get(), 8);
/// ```
#[cfg(feature = "cpu")]
#[must_use]
pub fn batch_search_width(
    point_count: usize,
    min_cluster_size: NonZeroUsize,
    hnsw_params: &crate::HnswParams,
) -> NonZeroUsize {
    let desired = min_cluster_size
        .get()
        .saturating_add(1)
        .max(hnsw_params.effective_ef_construction(point_count))
        .min(point_count);
    NonZeroUsize::new(desired).unwrap_or(NonZeroUsize::MIN)
}

/// Returns the guarded peak estimate when several searches can be live at once.
///
/// The batch core-distance stage runs one HNSW search per Rayon worker, so its
/// search-state allocation scales with the pool width rather than staying at
/// the single state assumed by
/// [`estimate_peak_bytes_for_hnsw_params`](crate::estimate_peak_bytes_for_hnsw_params).
/// Pass the width of the pool that will execute the run.
///
/// Each worker's search state is sized to [`batch_search_width`], which
/// follows the cluster size as well as the construction breadth. The
/// candidate-edge harvest stays on the construction breadth, because that is
/// what the build searches.
///
/// The estimate is still applied as a pre-flight guard: it predicts the run
/// from the configured parameters, cluster size, and concurrency, not from
/// observed allocation.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use chutoro_core::{
///     HnswParams, estimate_peak_bytes_for_hnsw_params,
///     estimate_peak_bytes_for_hnsw_params_with_concurrency,
/// };
///
/// let params = HnswParams::new(16, 64).expect("parameters must be valid");
/// let one_search = estimate_peak_bytes_for_hnsw_params(1_000, &params);
/// let eight_searches = estimate_peak_bytes_for_hnsw_params_with_concurrency(
///     1_000,
///     NonZeroUsize::new(2).expect("two is non-zero"),
///     &params,
///     NonZeroUsize::new(8).expect("eight is non-zero"),
/// );
/// assert!(eight_searches > one_search);
/// ```
#[cfg(feature = "cpu")]
#[must_use]
pub fn estimate_peak_bytes_for_hnsw_params_with_concurrency(
    point_count: usize,
    min_cluster_size: NonZeroUsize,
    hnsw_params: &crate::HnswParams,
    concurrent_searches: NonZeroUsize,
) -> u64 {
    let construction_width = hnsw_params.effective_ef_construction(point_count);
    estimate_peak_bytes_with_search_width(EstimateInputs {
        point_count,
        max_connections: hnsw_params.max_connections(),
        // Every worker holds one state sized to the width it searches at.
        effective_search_width: batch_search_width(point_count, min_cluster_size, hnsw_params)
            .get(),
        distance_cache_capacity: hnsw_params.distance_cache_config().max_entries().get(),
        // The build harvests at the construction breadth, not the batch width.
        harvest_width: construction_width,
        // The parameters own the sampler, so their expectation follows an
        // overridden level multiplier, including one that reaches the cap.
        layers_eighths: hnsw_params.expected_layers_eighths(),
        concurrent_searches: concurrent_searches.get(),
    })
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
