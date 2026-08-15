//! Batch-oracle helpers for the core-distance session tests.
//!
//! These helpers build the independent batch HNSW oracle, extract observed
//! core distances, and shape append orders so `core_distance.rs` stays
//! focused on the behaviours under test.

use std::{collections::BTreeSet, num::NonZeroUsize};

use crate::{ClusteringSession, CpuHnsw, DataSource, HnswError, HnswParams, Neighbour};

/// Builds an independent batch HNSW index to act as the oracle for recompute parity.
pub(super) fn expected_batch_cores<D: DataSource + Send + Sync>(
    source: &D,
    session: &ClusteringSession<D>,
) -> Result<Vec<f32>, HnswError> {
    let items = session.point_count();
    let params = session.config().hnsw_params().clone();
    let ef = expected_batch_ef(session.config().min_cluster_size(), &params, items);
    let index = CpuHnsw::build(source, params)?;

    (0..items)
        .map(|point| {
            let neighbours = index.search(source, point, ef)?;
            let others = neighbours
                .into_iter()
                .filter(|neighbour| neighbour.id != point)
                .collect::<Vec<_>>();
            Ok(expected_core_from_sorted_others(
                &others,
                session.config().min_cluster_size(),
            ))
        })
        .collect()
}

/// Selects the min-cluster neighbour, falling back to the last known neighbour or zero.
fn expected_core_from_sorted_others(
    neighbours: &[Neighbour],
    min_cluster_size: NonZeroUsize,
) -> f32 {
    neighbours
        .get(min_cluster_size.get() - 1)
        .or_else(|| neighbours.last())
        .map_or(0.0, |neighbour| neighbour.distance)
}

/// Mirrors batch EF: include the query slot, honour construction EF, and cap at item count.
fn expected_batch_ef(
    min_cluster_size: NonZeroUsize,
    hnsw_params: &HnswParams,
    items: usize,
) -> NonZeroUsize {
    let desired = min_cluster_size
        .get()
        .saturating_add(1)
        .max(hnsw_params.ef_construction())
        .min(items);
    NonZeroUsize::new(desired).unwrap_or(min_cluster_size)
}

/// Extracts every recomputed core distance from the session, or `None` when
/// any point still lacks one.
pub(super) fn observed_cores<D: DataSource + Send + Sync>(
    session: &ClusteringSession<D>,
) -> Option<Vec<f32>> {
    (0..session.point_count())
        .map(|point| session.core_distance(point))
        .collect()
}

/// Validates observed core distances against the batch oracle.
pub(super) fn assert_core_distances_eq<D: DataSource + Send + Sync>(
    session: &ClusteringSession<D>,
    expected: &[f32],
) {
    assert_eq!(observed_cores(session).as_deref(), Some(expected));
}

/// Sorts and deduplicates point indices for stable comparisons.
pub(super) fn unique_sorted(mut values: Vec<usize>) -> Vec<usize> {
    values.sort_unstable();
    values.dedup();
    values
}

/// Front-loads enough unique points so every prefixed point is saturated before tail appends.
pub(super) fn monotonic_append_order(min_cluster_size: usize, tail: Vec<usize>) -> Vec<usize> {
    let prefix = 0..=min_cluster_size;
    let mut seen = prefix.clone().collect::<BTreeSet<_>>();
    prefix
        .chain(tail.into_iter().filter(|index| seen.insert(*index)))
        .collect()
}
