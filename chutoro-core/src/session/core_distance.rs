//! Pure core-distance helpers for incremental clustering sessions.
//!
//! The helpers in this module make the domain decisions for core-distance
//! computation without knowing how HNSW searches are executed. Adapter code is
//! responsible for obtaining sorted neighbour lists and filtering out the query
//! point before calling these functions.

use std::{collections::BTreeSet, num::NonZeroUsize};

use super::ClusteringSession;
use crate::{DataSource, Neighbour, Result};
use tracing::{instrument, warn};

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
        neighbours[min_cluster_size.get() - 1].distance
    } else {
        neighbours
            .last()
            .map(|neighbour| neighbour.distance)
            .unwrap_or(0.0)
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
pub(super) fn recompute_targets(
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

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    pub(super) fn mark_core_distance_dirty(&mut self, index: usize) {
        let len = index.saturating_add(1);
        if self.core_distances.len() < len {
            self.core_distances.resize(len, f32::INFINITY);
            self.dirty_core_distances.resize(len, false);
        }
        self.core_distances[index] = f32::INFINITY;
        self.dirty_core_distances[index] = true;
    }

    fn inserted_core_distance_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.core_distances
            .iter()
            .zip(&self.dirty_core_distances)
            .enumerate()
            .filter_map(|(index, (distance, is_dirty))| {
                (*is_dirty || distance.is_finite()).then_some(index)
            })
    }

    fn dirty_core_distance_indices(&self) -> Vec<usize> {
        self.dirty_core_distances
            .iter()
            .enumerate()
            .filter_map(|(index, is_dirty)| is_dirty.then_some(index))
            .collect()
    }

    fn core_distance_ef(&self) -> Option<NonZeroUsize> {
        let point_count = NonZeroUsize::new(self.point_count())?;
        let ef_construction = NonZeroUsize::new(self.config.hnsw_params().ef_construction())?;
        Some(effective_ef(
            self.config.min_cluster_size(),
            ef_construction,
            point_count,
        ))
    }

    fn search_non_self_neighbours(&self, point: usize, ef: NonZeroUsize) -> Result<Vec<Neighbour>> {
        #[cfg(feature = "metrics")]
        metrics::counter!("chutoro.session.core_distance.queries_total").increment(1);

        let neighbours = self
            .index
            .search(self.source.as_ref(), point, ef)
            .map_err(|error| {
                let chutoro_error = self.map_hnsw_error(error);
                warn!(
                    point,
                    error = ?chutoro_error,
                    "core-distance recompute failed during HNSW search"
                );
                #[cfg(feature = "metrics")]
                metrics::counter!(
                    "chutoro.session.core_distance.errors_total",
                    "reason" => core_distance_error_reason(&chutoro_error)
                )
                .increment(1);
                chutoro_error
            })?;
        Ok(neighbours
            .into_iter()
            .filter(|neighbour| neighbour.id != point)
            .collect())
    }

    fn write_core_distance(&mut self, point: usize, neighbours: &[Neighbour]) {
        let core = core_distance_from_neighbours(neighbours, self.config.min_cluster_size());
        self.core_distances[point] = core;
        self.dirty_core_distances[point] = false;
    }

    /// Recomputes dirty and touched core distances.
    ///
    /// Newly appended points are always recomputed. Existing points are
    /// recomputed when they appear in a newly appended point's non-self
    /// neighbour list, which is the cheap incremental approximation used by
    /// roadmap item 11.1.4.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ChutoroError::DataSource`] when the backing source
    /// rejects a distance query. Returns
    /// [`crate::ChutoroError::CpuHnswFailure`] when HNSW search reports a
    /// structural failure.
    #[instrument(skip(self), level = "debug")]
    pub fn recompute_core_distances(&mut self) -> Result<()> {
        #[cfg(feature = "metrics")]
        let t0 = self.clock.now();

        let new_indices = self.dirty_core_distance_indices();
        if new_indices.is_empty() {
            #[cfg(feature = "metrics")]
            {
                metrics::histogram!("chutoro.session.core_distance.touched_existing_per_recompute")
                    .record(0.0);
                metrics::histogram!("chutoro.session.core_distance.recompute_seconds")
                    .record(self.clock.now().duration_since(t0).as_secs_f64());
            }
            return Ok(());
        }

        #[cfg(feature = "metrics")]
        metrics::counter!("chutoro.session.core_distance.appends_left_dirty_total").increment(1);

        let Some(ef) = self.core_distance_ef() else {
            return Ok(());
        };

        let mut neighbour_lists = Vec::with_capacity(new_indices.len());
        for &point in &new_indices {
            neighbour_lists.push(self.search_non_self_neighbours(point, ef)?);
        }

        let neighbour_slices = neighbour_lists
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let existing_targets = recompute_targets(&new_indices, &neighbour_slices);

        #[cfg(feature = "metrics")]
        metrics::histogram!("chutoro.session.core_distance.touched_existing_per_recompute")
            .record(existing_targets.len() as f64);

        for (point, neighbours) in new_indices.iter().copied().zip(&neighbour_lists) {
            self.write_core_distance(point, neighbours);
        }

        for point in existing_targets {
            let neighbours = self.search_non_self_neighbours(point, ef)?;
            self.write_core_distance(point, &neighbours);

            #[cfg(feature = "metrics")]
            metrics::counter!("chutoro.session.core_distance.recomputed_existing").increment(1);
        }

        #[cfg(feature = "metrics")]
        metrics::histogram!("chutoro.session.core_distance.recompute_seconds")
            .record(self.clock.now().duration_since(t0).as_secs_f64());

        Ok(())
    }

    /// Recomputes every inserted point's core distance.
    ///
    /// This mirrors the batch CPU pipeline's core-distance loop and clears all
    /// dirty cells that correspond to inserted points.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ChutoroError::DataSource`] when the backing source
    /// rejects a distance query. Returns
    /// [`crate::ChutoroError::CpuHnswFailure`] when HNSW search reports a
    /// structural failure.
    #[instrument(skip(self), level = "debug")]
    pub fn recompute_core_distances_full(&mut self) -> Result<()> {
        #[cfg(feature = "metrics")]
        let t0 = self.clock.now();

        let Some(ef) = self.core_distance_ef() else {
            return Ok(());
        };
        let indices = self.inserted_core_distance_indices().collect::<Vec<_>>();
        for point in indices {
            let neighbours = self.search_non_self_neighbours(point, ef)?;
            self.write_core_distance(point, &neighbours);
        }

        #[cfg(feature = "metrics")]
        metrics::histogram!("chutoro.session.core_distance.recompute_seconds")
            .record(self.clock.now().duration_since(t0).as_secs_f64());

        Ok(())
    }
}

#[cfg(feature = "metrics")]
fn core_distance_error_reason(error: &crate::ChutoroError) -> &'static str {
    match error {
        crate::ChutoroError::DataSource { .. } => "data_source",
        crate::ChutoroError::CpuHnswFailure { .. } => "hnsw_failure",
        _ => "other",
    }
}
