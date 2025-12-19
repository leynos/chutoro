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
mod forest;

use std::num::NonZeroUsize;

use crate::mst::MstEdge;

use self::condense::CondenseBuilder;

/// Errors returned by hierarchy extraction.
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum HierarchyError {
    /// Hierarchy extraction requires at least one node.
    #[error("cannot extract a hierarchy for an empty dataset")]
    EmptyDataset,
    /// The configured minimum cluster size exceeds the dataset size.
    #[error("min_cluster_size {min_cluster_size} exceeds node_count {node_count}")]
    MinClusterSizeTooLarge {
        /// Number of points in the dataset.
        node_count: usize,
        /// Minimum cluster size requested by the caller.
        min_cluster_size: usize,
    },
    /// An MST edge weight was invalid for hierarchy extraction.
    #[error("invalid MST edge weight {weight} for edge ({left}, {right})")]
    InvalidEdgeWeight {
        /// Smaller endpoint id for the offending edge.
        left: usize,
        /// Larger endpoint id for the offending edge.
        right: usize,
        /// Invalid weight value observed on the edge.
        weight: f32,
    },
}

impl HierarchyError {
    /// Returns a stable, machine-readable error code for the variant.
    #[must_use]
    pub const fn code(&self) -> HierarchyErrorCode {
        match self {
            Self::EmptyDataset => HierarchyErrorCode::EmptyDataset,
            Self::MinClusterSizeTooLarge { .. } => HierarchyErrorCode::MinClusterSizeTooLarge,
            Self::InvalidEdgeWeight { .. } => HierarchyErrorCode::InvalidEdgeWeight,
        }
    }
}

/// Machine-readable error codes for [`HierarchyError`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HierarchyErrorCode {
    /// The caller requested hierarchy extraction for an empty dataset.
    EmptyDataset,
    /// The configured minimum cluster size exceeds the dataset size.
    MinClusterSizeTooLarge,
    /// An input edge weight was invalid for hierarchy extraction.
    InvalidEdgeWeight,
}

impl HierarchyErrorCode {
    /// Returns the symbolic identifier for logging and metrics surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyDataset => "EMPTY_DATASET",
            Self::MinClusterSizeTooLarge => "MIN_CLUSTER_SIZE_TOO_LARGE",
            Self::InvalidEdgeWeight => "INVALID_EDGE_WEIGHT",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CondensedEvent {
    Point {
        index: usize,
        lambda: f32,
    },
    ChildCluster {
        cluster: usize,
        lambda: f32,
        size: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CondensedCluster {
    parent: Option<usize>,
    birth_lambda: f32,
    stability: f32,
    events: Vec<CondensedEvent>,
    children: Vec<usize>,
}

impl CondensedCluster {
    fn new(parent: Option<usize>, birth_lambda: f32) -> Self {
        Self {
            parent,
            birth_lambda,
            stability: 0.0,
            events: Vec::new(),
            children: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CondensedForest {
    clusters: Vec<CondensedCluster>,
    roots: Vec<usize>,
}

impl CondensedForest {
    fn validate_edges(edges: &[MstEdge]) -> Result<(), HierarchyError> {
        for edge in edges {
            let weight = edge.weight();
            if !weight.is_finite() || weight < 0.0 {
                return Err(HierarchyError::InvalidEdgeWeight {
                    left: edge.source(),
                    right: edge.target(),
                    weight,
                });
            }
        }
        Ok(())
    }

    fn process_root_into_condensed(
        root: usize,
        forest: &SingleLinkageForest,
        min_cluster_size: usize,
        condensed: &mut CondensedForest,
    ) {
        let root_size = forest.nodes[root].size;
        if root_size < min_cluster_size {
            // Entire component is below the minimum cluster size; it will
            // become noise during labelling.
            return;
        }

        let cluster_id = condensed.clusters.len();
        condensed.clusters.push(CondensedCluster::new(None, 0.0));
        condensed.roots.push(cluster_id);
        let mut builder = CondenseBuilder::new(forest, min_cluster_size, &mut condensed.clusters);
        builder.condense_cluster(root, cluster_id);
    }

    pub(crate) fn from_mst(
        node_count: usize,
        edges: &[MstEdge],
        min_cluster_size: NonZeroUsize,
    ) -> Result<Self, HierarchyError> {
        let min_cluster_size = min_cluster_size.get();
        if node_count == 0 {
            return Err(HierarchyError::EmptyDataset);
        }
        if min_cluster_size > node_count {
            return Err(HierarchyError::MinClusterSizeTooLarge {
                node_count,
                min_cluster_size,
            });
        }

        Self::validate_edges(edges)?;

        let forest = SingleLinkageForest::from_mst(node_count, edges);
        let mut condensed = Self {
            clusters: Vec::new(),
            roots: Vec::new(),
        };

        for root in forest.roots.iter().copied() {
            Self::process_root_into_condensed(root, &forest, min_cluster_size, &mut condensed);
        }

        Ok(condensed)
    }
}

pub(crate) fn extract_flat_labels(
    node_count: usize,
    condensed: &CondensedForest,
) -> Result<Vec<usize>, HierarchyError> {
    if node_count == 0 {
        return Err(HierarchyError::EmptyDataset);
    }

    if condensed.clusters.is_empty() {
        // No condensed clusters were retained; treat all points as noise with
        // noise label `0`.
        return Ok(vec![0; node_count]);
    }

    let selected = select_stable_clusters(condensed);
    let mut selected_ids: Vec<usize> = selected.into_iter().collect();
    selected_ids.sort_unstable();

    let mut label_lookup = vec![None; condensed.clusters.len()];
    for (label, cluster_id) in selected_ids.iter().copied().enumerate() {
        label_lookup[cluster_id] = Some(label);
    }

    let mut labels = vec![None; node_count];
    let mut labeller = Labeller::new(condensed, &label_lookup, &mut labels);
    for root in condensed.roots.iter().copied() {
        labeller.label_cluster(root, None);
    }

    let cluster_count = selected_ids.len();
    Ok(labels
        .into_iter()
        .map(|label| label.unwrap_or(cluster_count))
        .collect())
}

struct Labeller<'a> {
    condensed: &'a CondensedForest,
    label_lookup: &'a [Option<usize>],
    labels: &'a mut [Option<usize>],
}

impl<'a> Labeller<'a> {
    fn new(
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

    fn label_cluster(&mut self, cluster_id: usize, inherited: Option<usize>) {
        let cluster_label = self.label_lookup[cluster_id].or(inherited);
        let cluster = &self.condensed.clusters[cluster_id];

        for event in &cluster.events {
            match *event {
                CondensedEvent::Point { index, .. } => {
                    self.labels[index] = cluster_label;
                }
                CondensedEvent::ChildCluster { cluster, .. } => {
                    self.label_cluster(cluster, cluster_label);
                }
            }
        }
    }
}

fn select_stable_clusters(condensed: &CondensedForest) -> Vec<usize> {
    let mut selected = Vec::new();
    for root in condensed.roots.iter().copied() {
        select_stable_clusters_inner(condensed, root, &mut selected);
    }
    if selected.is_empty() {
        // Fallback: select all roots to avoid returning only noise for
        // well-formed condensed forests.
        selected.extend(condensed.roots.iter().copied());
    }
    selected
}

fn select_stable_clusters_inner(
    condensed: &CondensedForest,
    cluster_id: usize,
    selected: &mut Vec<usize>,
) -> f32 {
    let cluster = &condensed.clusters[cluster_id];
    if cluster.children.is_empty() {
        selected.push(cluster_id);
        return cluster.stability;
    }

    let mut child_score = 0.0_f32;
    let mut child_selected = Vec::with_capacity(cluster.children.len());
    for child in &cluster.children {
        let before = selected.len();
        let score = select_stable_clusters_inner(condensed, *child, selected);
        child_score += score;
        child_selected.push((before, selected.len()));
    }

    if child_score > cluster.stability {
        return child_score;
    }

    // Replace child selections with the current cluster.
    // Drain child selections in reverse order to preserve valid indices.
    for (start, end) in child_selected.into_iter().rev() {
        selected.drain(start..end);
    }
    selected.push(cluster_id);
    cluster.stability
}

#[derive(Clone, Debug)]
struct LinkageNode {
    left: Option<usize>,
    right: Option<usize>,
    weight: f32,
    size: usize,
    point: Option<usize>,
}

#[derive(Clone, Debug)]
struct SingleLinkageForest {
    nodes: Vec<LinkageNode>,
    roots: Vec<usize>,
}
