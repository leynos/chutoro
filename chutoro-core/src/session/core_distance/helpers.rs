//! Pure helper functions shared by the core-distance recompute paths.

use std::{collections::BTreeSet, num::NonZeroUsize};

use crate::Neighbour;

/// Computes a core distance from sorted non-self neighbours.
///
/// `neighbours` must already be sorted in ascending distance order and must
/// not contain the query point. The function mirrors the batch CPU pipeline:
/// return the `m - 1` element when at least `m` neighbours exist, the last
/// available neighbour when the list is shorter, or `0.0` when the list is
/// empty.
pub(super) fn core_distance_from_neighbours(
    neighbours: &[Neighbour],
    min_cluster_size: NonZeroUsize,
) -> f32 {
    if neighbours.len() >= min_cluster_size.get() {
        neighbours
            .get(min_cluster_size.get().saturating_sub(1))
            .map_or(0.0, |neighbour| neighbour.distance)
    } else {
        neighbours
            .last()
            .map_or(0.0, |neighbour| neighbour.distance)
    }
}

/// Computes the HNSW `ef` used for core-distance neighbour searches.
///
/// This mirrors the batch path's
/// `max(min_cluster_size + 1, ef_construction).min(point_count)` rule.
pub(super) fn effective_ef(
    min_cluster_size: NonZeroUsize,
    ef_construction: NonZeroUsize,
    point_count: NonZeroUsize,
) -> NonZeroUsize {
    let desired = min_cluster_size
        .get()
        .saturating_add(1)
        .max(ef_construction.get())
        .min(point_count.get());
    NonZeroUsize::new(desired).unwrap_or(point_count)
}

/// Computes existing points touched by newly-inserted points.
///
/// The returned vector is sorted, deduplicated, and excludes every source
/// index present in `new_indices`.
pub(crate) fn recompute_targets(
    new_indices: &[usize],
    neighbour_lists: &[&[Neighbour]],
) -> Vec<usize> {
    let new = new_indices.iter().copied().collect::<BTreeSet<_>>();
    neighbour_lists
        .iter()
        .flat_map(|neighbours| neighbours.iter().map(|neighbour| neighbour.id))
        .filter(|index| !new.contains(index))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Derive core distances for `points` from their pre-computed neighbours.
pub(super) fn core_updates_for(
    points: &[usize],
    neighbour_lists: &[Vec<Neighbour>],
    min_cluster_size: NonZeroUsize,
) -> Vec<(usize, f32)> {
    let mut updates = Vec::with_capacity(points.len());
    for (point, neighbours) in points.iter().copied().zip(neighbour_lists) {
        updates.push((
            point,
            core_distance_from_neighbours(neighbours, min_cluster_size),
        ));
    }
    updates
}

/// Classify a core-distance failure for low-cardinality metrics.
#[cfg(feature = "metrics")]
pub(super) const fn core_distance_error_reason(error: &crate::ChutoroError) -> &'static str {
    match error {
        crate::ChutoroError::DataSource { .. } => "data_source",
        crate::ChutoroError::CpuHnswFailure { .. } => "hnsw_failure",
        _ => "other",
    }
}
