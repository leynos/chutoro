//! Neighbour collection for incremental core-distance recomputation.
//!
//! `recompute_core_distances` decides *when* core distances are recomputed;
//! this module performs the searches that supply them. Each helper owns one
//! stage of that work — the newly inserted points, the existing points their
//! arrival disturbs, and the core-distance extraction over each — so the
//! driver reads as a sequence of stages rather than three nested loops.

use std::num::NonZeroUsize;

use super::ClusteringSession;
use crate::{DataSource, Neighbour, Result, core_distance::core_distance_from_neighbours};

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    /// Searches the non-self neighbours of every newly inserted point.
    pub(super) fn new_point_neighbour_lists(
        &self,
        new_indices: &[usize],
        ef: NonZeroUsize,
    ) -> Result<Vec<Vec<Neighbour>>> {
        let mut neighbour_lists = Vec::with_capacity(new_indices.len());
        for &point in new_indices {
            neighbour_lists.push(self.search_non_self_neighbours(point, ef)?);
        }
        Ok(neighbour_lists)
    }

    /// Derives the core distance of each newly inserted point.
    ///
    /// The neighbour lists are consumed by reference: their slices have already
    /// been used to choose the existing points to revisit.
    pub(super) fn new_point_core_distances(
        &self,
        new_indices: &[usize],
        neighbour_lists: &[Vec<Neighbour>],
    ) -> Vec<(usize, f32)> {
        let min_cluster_size = self.config.min_cluster_size();
        new_indices
            .iter()
            .copied()
            .zip(neighbour_lists)
            .map(|(point, neighbours)| {
                (
                    point,
                    core_distance_from_neighbours(neighbours, min_cluster_size),
                )
            })
            .collect()
    }

    /// Recomputes the core distance of each existing point the insert disturbed.
    ///
    /// The search is repeated rather than reused because an existing point's
    /// neighbour list was not collected by this pass.
    pub(super) fn existing_point_core_distances(
        &self,
        existing_targets: &[usize],
        ef: NonZeroUsize,
    ) -> Result<Vec<(usize, f32)>> {
        let min_cluster_size = self.config.min_cluster_size();
        let mut updates = Vec::with_capacity(existing_targets.len());
        for &point in existing_targets {
            let neighbours = self.search_non_self_neighbours(point, ef)?;
            updates.push((
                point,
                core_distance_from_neighbours(&neighbours, min_cluster_size),
            ));
        }
        Ok(updates)
    }
}
