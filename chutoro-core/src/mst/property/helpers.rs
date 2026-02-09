//! Shared helper functions for MST property-based tests.
//!
//! Provides common utilities used across multiple property modules,
//! including union-find operations and weight accumulation.

use crate::MstEdge;

/// Path-compressing find for union-find verification.
pub(super) fn find_root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

/// Sums edge weights as `f64` for lossless accumulation.
pub(super) fn total_weight_f64(edges: &[MstEdge]) -> f64 {
    edges.iter().map(|e| f64::from(e.weight())).sum()
}

/// Returns `true` when an edge should be excluded from MST consideration.
///
/// An edge is invalid when any of the following hold:
/// - it is a self-loop,
/// - either endpoint falls outside the node range, or
/// - its weight is non-finite (NaN / infinity).
pub(super) fn is_invalid_edge(
    source: usize,
    target: usize,
    node_count: usize,
    weight: f32,
) -> bool {
    let is_self_loop = source == target;
    let is_out_of_bounds = source >= node_count || target >= node_count;
    let is_non_finite = !weight.is_finite();
    is_self_loop || is_out_of_bounds || is_non_finite
}
