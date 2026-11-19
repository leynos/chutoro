//! Shared fixtures for the parallel batch core-distance tests.
//!
//! The fixtures build deterministic indices and compute the sequential
//! reference, so every case compares the parallel stage against the same
//! baseline instead of restating its setup. Nothing here is production code.

use std::num::NonZeroUsize;

use proptest::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::test_utils::NumericTestSource;
use crate::{CpuHnsw, DataSource, HnswParams, Neighbour};

use super::super::CoreDistanceInputs;

/// Runs `task` inside a Rayon pool with exactly `threads` workers.
///
/// Building the pool can fail only for an unusable thread count or an operating
/// system resource failure, so callers decide how to report that.
pub(super) fn with_pool<T: Send>(
    threads: usize,
    task: impl FnOnce() -> T + Send,
) -> Result<T, rayon::ThreadPoolBuildError> {
    let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
    Ok(pool.install(task))
}

/// Builds `count` distinct, evenly spaced fixture values.
#[expect(
    clippy::cast_precision_loss,
    reason = "the fixture is a handwritten sequence of small whole numbers"
)]
pub(super) fn distinct_values(count: usize) -> Vec<f32> {
    (0..count).map(|value| value as f32).collect()
}

/// Builds a deterministic index over `source` and the `ef` its searches use.
///
/// The `ef` mirrors the production formula for this stage so the tests
/// exercise the same search width the pipeline requests.
pub(super) fn build_fixture_index(
    source: &NumericTestSource,
) -> Result<(CpuHnsw, NonZeroUsize), crate::HnswError> {
    let params = HnswParams::new(4, 16)?.with_rng_seed(42);
    let (index, _) = CpuHnsw::build_with_edges(source, params)?;
    // `max(1)` keeps the search width non-zero for an empty source.
    let ef = NonZeroUsize::new(source.len().max(1)).unwrap_or(NonZeroUsize::MIN);
    Ok((index, ef))
}

/// Computes core distances serially, restating the selection rule directly.
///
/// This deliberately does not call the shared helper: it is the independent
/// reference the parallel path is compared against.
pub(super) fn sequential_core_distances<D: DataSource + Sync>(
    source: &D,
    index: &CpuHnsw,
    inputs: &CoreDistanceInputs,
) -> Result<Vec<f32>, crate::HnswError> {
    let mut core_distances = Vec::with_capacity(inputs.items);
    for point in 0..inputs.items {
        let neighbours = index.search(source, point, inputs.ef)?;
        let others: Vec<Neighbour> = neighbours
            .into_iter()
            .filter(|neighbour| neighbour.id != point)
            .collect();
        let core = if others.len() >= inputs.min_cluster_size.get() {
            others
                .get(inputs.min_cluster_size.get().saturating_sub(1))
                .map_or(0.0, |neighbour| neighbour.distance)
        } else {
            others.last().map_or(0.0, |neighbour| neighbour.distance)
        };
        core_distances.push(core);
    }
    Ok(core_distances)
}

/// Generates a source's values and a plausible minimum cluster size.
pub(super) fn generated_source() -> impl Strategy<Value = (Vec<f32>, usize)> {
    (2usize..10)
        .prop_flat_map(|point_count| (prop::collection::vec(0.0f32..16.0, point_count), 1usize..6))
}
