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
