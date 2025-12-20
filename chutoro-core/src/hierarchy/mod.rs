//! Hierarchy extraction from the mutual-reachability minimum spanning tree.
//!
//! This module implements the final stage of the CPU FISHDBC pipeline:
//! extracting a flat clustering from the mutual-reachability MST by:
//!
//! - Deriving a single-linkage hierarchy from the MST (equivalent to the MST
//!   dendrogram).
//! - Condensing the hierarchy with `min_cluster_size` (HDBSCAN-style
//!   condensation where a cluster can "continue" when one branch is too
//!   small).
//! - Scoring clusters by stability ("excess of mass") and selecting the most
//!   persistent clusters.
//! - Labelling points by the selected cluster they are contained within,
//!   falling back to a dedicated noise label when no selected cluster applies.
//!
//! The implementation is intentionally sequential to keep the logic simple and
//! deterministic. This stage is typically not the dominant runtime cost
//! relative to HNSW construction and MST computation.

mod single_linkage;
mod union_find;

use std::num::NonZeroUsize;

use crate::mst::MstEdge;

pub use self::single_linkage::{HierarchyError, HierarchyErrorCode};

use self::single_linkage::{CondensedForest, extract_flat_labels};

/// Configuration for hierarchy extraction.
#[derive(Debug, Clone, Copy)]
pub struct HierarchyConfig {
    min_cluster_size: NonZeroUsize,
}

impl HierarchyConfig {
    /// Creates a configuration using the provided `min_cluster_size`.
    #[must_use]
    pub fn new(min_cluster_size: NonZeroUsize) -> Self {
        Self { min_cluster_size }
    }

    /// Returns the minimum cluster size.
    #[must_use]
    pub fn min_cluster_size(&self) -> NonZeroUsize {
        self.min_cluster_size
    }
}

/// Extracts flat cluster labels from a mutual-reachability MST/forest.
///
/// The input edges must describe a minimum spanning forest over `node_count`
/// points, where each edge weight is a mutual-reachability distance.
///
/// The output labels are contiguous `usize` identifiers starting at `0`. When
/// any point is classified as noise, an additional label is appended so noise
/// points receive a valid contiguous identifier as well. When all points are
/// classified as noise, no cluster labels are emitted and the noise label is
/// `0`.
///
/// # Errors
/// Returns [`HierarchyError`] when `node_count == 0`, when `min_cluster_size`
/// is larger than `node_count`, or when the edge weights are invalid (negative
/// or non-finite).
pub fn extract_labels_from_mst(
    node_count: usize,
    edges: &[MstEdge],
    config: HierarchyConfig,
) -> Result<Vec<usize>, HierarchyError> {
    let condensed = CondensedForest::from_mst(node_count, edges, config.min_cluster_size())?;
    extract_flat_labels(node_count, &condensed)
}

#[cfg(test)]
mod tests;
