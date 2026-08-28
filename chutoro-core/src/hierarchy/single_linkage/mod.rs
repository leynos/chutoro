//! Single-linkage hierarchy construction and HDBSCAN-style condensation.
//!
//! The mutual-reachability MST encodes the same single-linkage hierarchy as
//! the full mutual-reachability graph. We recover that hierarchy by sorting
//! the MST edges in non-decreasing weight order and performing a union-find
//! merge, producing a dendrogram (a binary tree per connected component).
//!
//! We then "condense" the hierarchy using `min_cluster_size`, following the
//! HDBSCAN condensed tree procedure:
//!
//! - When a cluster would split into two subclusters that both satisfy
//!   `min_cluster_size`, we emit two child clusters and terminate the parent.
//! - When only one branch satisfies `min_cluster_size`, the cluster continues
//!   down the large branch with the same cluster id, and the points in the
//!   small branch are emitted as point leaves at the split lambda.
//! - When neither branch satisfies `min_cluster_size`, the cluster terminates
//!   and all remaining points are emitted as point leaves.
//!
//! This yields a condensed forest suitable for computing stability scores and
//! extracting a flat clustering.

mod condense;
mod error;
mod forest;

use core::ops::AddAssign;
use std::num::NonZeroUsize;

use crate::mst::MstEdge;

use self::condense::CondenseBuilder;
pub use self::error::{HierarchyError, HierarchyErrorCode};

/// Event emitted while condensing a linkage-tree cluster.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CondensedEvent {
    /// A data point leaves the current cluster.
    Point {
        /// Original data-point identifier.
        index: usize,
        /// Lifetime at which the point leaves.
        lambda: f32,
    },
    /// A new child cluster begins at the split.
    ChildCluster {
        /// Identifier assigned to the child cluster.
        cluster: usize,
        /// Lifetime at which the child begins.
        lambda: f32,
        /// Number of points assigned to the child.
        size: usize,
    },
}

/// One cluster in the condensed hierarchy and its stability evidence.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CondensedCluster {
    /// Parent cluster, if this is not a root.
    parent: Option<usize>,
    /// Lifetime at which this cluster was created.
    birth_lambda: f32,
    /// Accumulated cluster lifetime weighted by membership.
    stability: f32,
    /// Point and child-cluster transitions emitted by this cluster.
    events: Vec<CondensedEvent>,
    /// Direct child-cluster identifiers.
    children: Vec<usize>,
}

impl CondensedCluster {
    /// Create an empty cluster with its parent and creation lifetime.
    const fn new(parent: Option<usize>, birth_lambda: f32) -> Self {
        Self {
            parent,
            birth_lambda,
            stability: 0.0,
            events: Vec::new(),
            children: Vec::new(),
        }
    }
}

/// Collection of condensed clusters and their component roots.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CondensedForest {
    /// All clusters in creation order.
    clusters: Vec<CondensedCluster>,
    /// Root clusters, one per retained connected component.
    roots: Vec<usize>,
}

impl CondensedForest {
    /// Reject an MST endpoint that lies outside the input dataset.
    const fn validate_endpoint(endpoint: usize, node_count: usize) -> Result<(), HierarchyError> {
        if endpoint < node_count {
            return Ok(());
        }

        Err(HierarchyError::InvalidEdgeEndpoint {
            endpoint,
            node_count,
        })
    }

    /// Validate endpoints and finite, non-negative weights for MST edges.
    fn validate_edges(node_count: usize, edges: &[MstEdge]) -> Result<(), HierarchyError> {
        for edge in edges {
            for endpoint in [edge.source(), edge.target()] {
                Self::validate_endpoint(endpoint, node_count)?;
            }
            let weight = edge.weight();
            if !weight.is_finite() || weight < 0.0 {
                let left = edge.source().min(edge.target());
                let right = edge.source().max(edge.target());
                return Err(HierarchyError::InvalidEdgeWeight {
                    left,
                    right,
                    weight,
                });
            }
        }
        Ok(())
    }

    /// Condense one sufficiently large linkage-tree component into `condensed`.
    fn process_root_into_condensed(
        root: usize,
        forest: &SingleLinkageForest,
        min_cluster_size: usize,
        condensed: &mut Self,
    ) -> Result<(), HierarchyError> {
        let root_size = forest
            .nodes
            .get(root)
            .ok_or(HierarchyError::InvalidForestReference { node_id: root })?
            .size;
        if root_size < min_cluster_size {
            // Entire component is below the minimum cluster size; it will
            // become noise during labelling.
            return Ok(());
        }

        let cluster_id = condensed.clusters.len();
        condensed.clusters.push(CondensedCluster::new(None, 0.0));
        condensed.roots.push(cluster_id);
        let mut builder = CondenseBuilder::new(forest, min_cluster_size, &mut condensed.clusters);
        builder.condense_cluster(root, cluster_id)
    }

    /// Build a condensed forest from a validated mutual-reachability MST.
    pub(crate) fn from_mst(
        node_count: usize,
        edges: &[MstEdge],
        minimum_cluster_size: NonZeroUsize,
    ) -> Result<Self, HierarchyError> {
        let min_cluster_size = minimum_cluster_size.get();
        if node_count == 0 {
            return Err(HierarchyError::EmptyDataset);
        }
        if min_cluster_size > node_count {
            return Err(HierarchyError::MinClusterSizeTooLarge {
                node_count,
                min_cluster_size,
            });
        }

        Self::validate_edges(node_count, edges)?;

        let forest = SingleLinkageForest::from_mst(node_count, edges)?;
        let mut condensed = Self {
            clusters: Vec::new(),
            roots: Vec::new(),
        };

        for root in forest.roots.iter().copied() {
            Self::process_root_into_condensed(root, &forest, min_cluster_size, &mut condensed)?;
        }

        Ok(condensed)
    }
}

/// Extracts flat cluster labels from a condensed hierarchy forest.
///
/// Cluster labels are contiguous `usize` identifiers starting at `0`. Noise
/// points are assigned a dedicated label appended after the selected clusters.
///
/// When no clusters are selected (for example when all components are smaller
/// than `min_cluster_size` during condensation), the noise label is `0`.
pub(crate) fn extract_flat_labels(
    node_count: usize,
    condensed: &CondensedForest,
) -> Result<Vec<usize>, HierarchyError> {
    if node_count == 0 {
        return Err(HierarchyError::EmptyDataset);
    }
    if condensed.clusters.is_empty() {
        // No condensed clusters implies every point is noise.
        return Ok(vec![0; node_count]);
    }

    let selected = select_stable_clusters(condensed)?;
    let mut selected_ids: Vec<usize> = selected.into_iter().collect();
    selected_ids.sort_unstable();

    let mut label_lookup = vec![None; condensed.clusters.len()];
    for (label, cluster_id) in selected_ids.iter().copied().enumerate() {
        *label_lookup
            .get_mut(cluster_id)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id })? = Some(label);
    }

    let mut labels = vec![None; node_count];
    let mut labeller = Labeller::new(condensed, &label_lookup, &mut labels);
    for root in condensed.roots.iter().copied() {
        labeller.label_cluster(root, None)?;
    }

    let cluster_count = selected_ids.len();
    // When `cluster_count == 0`, the returned labels are all `0`, representing
    // the dedicated noise label (there are no clusters to offset it from).
    Ok(labels
        .into_iter()
        .map(|label| label.unwrap_or(cluster_count))
        .collect())
}

/// Assigns selected condensed clusters to their point-leaf events.
struct Labeller<'a> {
    /// Condensed hierarchy to traverse.
    condensed: &'a CondensedForest,
    /// Selected label for each cluster, if selected.
    label_lookup: &'a [Option<usize>],
    /// Output label slot for each input point.
    labels: &'a mut [Option<usize>],
}

impl<'a> Labeller<'a> {
    /// Create a labeller over the selected clusters and output slots.
    const fn new(
        condensed: &'a CondensedForest,
        label_lookup: &'a [Option<usize>],
        labels: &'a mut [Option<usize>],
    ) -> Self {
        Self {
            condensed,
            label_lookup,
            labels,
        }
    }

    /// Label a cluster's point events and recursively visit its children.
    fn label_cluster(
        &mut self,
        cluster_id: usize,
        inherited: Option<usize>,
    ) -> Result<(), HierarchyError> {
        let cluster_label = self
            .label_lookup
            .get(cluster_id)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?
            .or(inherited);
        let cluster = self
            .condensed
            .clusters
            .get(cluster_id)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?;

        for event in &cluster.events {
            match *event {
                CondensedEvent::Point { index, .. } => {
                    let node_count = self.labels.len();
                    *self
                        .labels
                        .get_mut(index)
                        .ok_or(HierarchyError::InvalidPointReference {
                            point_id: index,
                            node_count,
                        })? = cluster_label;
                }
                CondensedEvent::ChildCluster {
                    cluster: child_cluster,
                    ..
                } => {
                    self.label_cluster(child_cluster, cluster_label)?;
                }
            }
        }

        Ok(())
    }
}

/// Select the maximum-stability cluster set beneath every condensed root.
fn select_stable_clusters(condensed: &CondensedForest) -> Result<Vec<usize>, HierarchyError> {
    let mut selected = Vec::new();
    for root in condensed.roots.iter().copied() {
        select_stable_clusters_inner(condensed, root, &mut selected)?;
    }
    if selected.is_empty() {
        // Fallback: select all roots to avoid returning only noise for
        // well-formed condensed forests.
        selected.extend(condensed.roots.iter().copied());
    }
    Ok(selected)
}

/// Select stable clusters recursively and return the best subtree score.
fn select_stable_clusters_inner(
    condensed: &CondensedForest,
    cluster_id: usize,
    selected: &mut Vec<usize>,
) -> Result<f32, HierarchyError> {
    let cluster = condensed
        .clusters
        .get(cluster_id)
        .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?;
    if cluster.children.is_empty() {
        selected.push(cluster_id);
        return Ok(cluster.stability);
    }

    let mut child_score = 0.0_f32;
    let mut child_selected = Vec::with_capacity(cluster.children.len());
    for child in &cluster.children {
        let before = selected.len();
        let score = select_stable_clusters_inner(condensed, *child, selected)?;
        child_score.add_assign(score);
        child_selected.push((before, selected.len()));
    }

    if child_score > cluster.stability {
        return Ok(child_score);
    }

    // Replace child selections with the current cluster.
    // Drain child selections in reverse order to preserve valid indices.
    for (start, end) in child_selected.into_iter().rev() {
        selected.drain(start..end);
    }
    selected.push(cluster_id);
    Ok(cluster.stability)
}

#[derive(Clone, Debug)]
/// One node in the binary linkage dendrogram.
struct LinkageNode {
    /// Left child node for an internal branch.
    left: Option<usize>,
    /// Right child node for an internal branch.
    right: Option<usize>,
    /// MST weight at which the children merge.
    weight: f32,
    /// Number of point leaves beneath this node.
    size: usize,
    /// Input point represented by a leaf node.
    point: Option<usize>,
}

// `SingleLinkageForest::from_mst` lives in the `forest` submodule.
/// Linkage nodes and the roots of their disconnected components.
#[derive(Clone, Debug)]
struct SingleLinkageForest {
    /// Linkage nodes in leaf-first construction order.
    nodes: Vec<LinkageNode>,
    /// Root node of each disconnected component.
    roots: Vec<usize>,
}
