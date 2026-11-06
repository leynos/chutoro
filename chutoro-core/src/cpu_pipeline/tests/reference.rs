//! Independent sequential reference for the mutual-reachability transform.
//!
//! The reference keeps the pre-change shape of the transform: walk the harvest
//! directly, weight each edge in arrival order, and collect. Comparing against
//! it isolates the parallel slice walk and the ordering it produces from every
//! other pipeline stage.

use crate::{CandidateEdge, EdgeHarvest};

/// Applies mutual-reachability weighting to every harvested edge in order.
pub(super) fn sequential_mutual_edges(
    harvest: &EdgeHarvest,
    core_distances: &[f32],
) -> Vec<CandidateEdge> {
    harvest
        .iter()
        .map(|edge| {
            let left = edge.source();
            let right = edge.target();
            let left_core_distance = core_distances
                .get(left)
                .copied()
                .expect("every fixture endpoint has a core distance");
            let right_core_distance = core_distances
                .get(right)
                .copied()
                .expect("every fixture endpoint has a core distance");
            let weight = edge
                .distance()
                .max(left_core_distance)
                .max(right_core_distance);

            CandidateEdge::new(left, right, weight, edge.sequence())
        })
        .collect()
}