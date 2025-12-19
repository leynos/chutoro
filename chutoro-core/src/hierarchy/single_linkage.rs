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

use std::num::NonZeroUsize;

use crate::mst::MstEdge;

#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
/// Errors returned by hierarchy extraction.
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

    let mut out = Vec::with_capacity(node_count);
    let cluster_count = selected_ids.len();
    for label in labels {
        match label {
            Some(cluster_label) => out.push(cluster_label),
            None => {
                out.push(cluster_count);
            }
        }
    }

    Ok(out)
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
    let mut child_selected = Vec::new();
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

impl SingleLinkageForest {
    fn merge_edges(
        dsu: &mut DisjointSet,
        nodes: &mut Vec<LinkageNode>,
        edges_sorted: Vec<MstEdge>,
    ) {
        for edge in edges_sorted {
            let left_root = dsu.find(edge.source());
            let right_root = dsu.find(edge.target());
            if left_root == right_root {
                continue;
            }
            let left_node = dsu.component_node[left_root];
            let right_node = dsu.component_node[right_root];
            let new_id = nodes.len();
            let size = nodes[left_node].size + nodes[right_node].size;
            nodes.push(LinkageNode {
                left: Some(left_node),
                right: Some(right_node),
                weight: edge.weight(),
                size,
                point: None,
            });
            let merged = dsu.union(left_root, right_root);
            dsu.component_node[merged] = new_id;
        }
    }

    fn collect_roots(dsu: &mut DisjointSet, node_count: usize) -> Vec<usize> {
        let mut roots: Vec<usize> = (0..node_count)
            .filter_map(|node| {
                let root = dsu.find(node);
                (root == node).then_some(dsu.component_node[root])
            })
            .collect();

        roots.sort_unstable();
        roots.dedup();
        roots
    }

    fn from_mst(node_count: usize, edges: &[MstEdge]) -> Self {
        let mut nodes = Vec::with_capacity(node_count.saturating_mul(2).saturating_sub(1));
        for point in 0..node_count {
            nodes.push(LinkageNode {
                left: None,
                right: None,
                weight: 0.0,
                size: 1,
                point: Some(point),
            });
        }

        let mut dsu = DisjointSet::new(node_count);
        let mut edges_sorted = edges.to_vec();
        edges_sorted.sort_unstable();

        Self::merge_edges(&mut dsu, &mut nodes, edges_sorted);
        let roots = Self::collect_roots(&mut dsu, node_count);
        Self { nodes, roots }
    }
}

struct CondenseBuilder<'a> {
    forest: &'a SingleLinkageForest,
    min_cluster_size: usize,
    clusters: &'a mut Vec<CondensedCluster>,
}

impl<'a> CondenseBuilder<'a> {
    fn new(
        forest: &'a SingleLinkageForest,
        min_cluster_size: usize,
        clusters: &'a mut Vec<CondensedCluster>,
    ) -> Self {
        Self {
            forest,
            min_cluster_size,
            clusters,
        }
    }

    fn condense_cluster(&mut self, node_id: usize, cluster_id: usize) {
        let node = &self.forest.nodes[node_id];
        let Some((left, right)) = node.left.zip(node.right) else {
            if let Some(point) = node.point {
                record_point_event(self.clusters, cluster_id, point, f32::INFINITY);
            }
            return;
        };

        let lambda = weight_to_lambda(node.weight);
        let left_size = self.forest.nodes[left].size;
        let right_size = self.forest.nodes[right].size;
        let left_big = left_size >= self.min_cluster_size;
        let right_big = right_size >= self.min_cluster_size;

        match (left_big, right_big) {
            (true, true) => {
                let left_cluster = self.create_child_cluster(cluster_id, lambda, left_size);
                let right_cluster = self.create_child_cluster(cluster_id, lambda, right_size);
                self.condense_cluster(left, left_cluster);
                self.condense_cluster(right, right_cluster);
            }
            (true, false) => {
                self.emit_pruned_points(right, cluster_id, lambda);
                self.condense_cluster(left, cluster_id);
            }
            (false, true) => {
                self.emit_pruned_points(left, cluster_id, lambda);
                self.condense_cluster(right, cluster_id);
            }
            (false, false) => {
                self.emit_pruned_points(left, cluster_id, lambda);
                self.emit_pruned_points(right, cluster_id, lambda);
            }
        }
    }

    fn create_child_cluster(&mut self, parent: usize, lambda: f32, size: usize) -> usize {
        let child_id = self.clusters.len();
        self.clusters
            .push(CondensedCluster::new(Some(parent), lambda));
        self.clusters[parent].children.push(child_id);
        self.clusters[parent]
            .events
            .push(CondensedEvent::ChildCluster {
                cluster: child_id,
                lambda,
                size,
            });
        record_stability_increment(&mut self.clusters[parent], lambda, size as f32);
        child_id
    }

    fn emit_pruned_points(&mut self, node_id: usize, cluster_id: usize, lambda: f32) {
        let mut stack = vec![node_id];
        while let Some(current) = stack.pop() {
            let node = &self.forest.nodes[current];
            if let Some(point) = node.point {
                record_point_event(self.clusters, cluster_id, point, lambda);
                continue;
            }
            if let Some(left) = node.left {
                stack.push(left);
            }
            if let Some(right) = node.right {
                stack.push(right);
            }
        }
    }
}

fn record_point_event(
    clusters: &mut [CondensedCluster],
    cluster_id: usize,
    point: usize,
    lambda: f32,
) {
    let cluster = &mut clusters[cluster_id];
    cluster.events.push(CondensedEvent::Point {
        index: point,
        lambda,
    });
    record_stability_increment(cluster, lambda, 1.0);
}

fn record_stability_increment(cluster: &mut CondensedCluster, lambda: f32, size: f32) {
    let increment = (lambda - cluster.birth_lambda) * size;
    cluster.stability += increment;
}

fn weight_to_lambda(weight: f32) -> f32 {
    if weight == 0.0 {
        f32::INFINITY
    } else {
        1.0 / weight
    }
}

#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
    component_node: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            component_node: (0..n).collect(),
        }
    }

    fn find(&mut self, node: usize) -> usize {
        let parent = self.parent[node];
        if parent == node {
            return node;
        }
        let root = self.find(parent);
        self.parent[node] = root;
        root
    }

    fn union(&mut self, left: usize, right: usize) -> usize {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return left;
        }
        let left_rank = self.rank[left];
        let right_rank = self.rank[right];
        if left_rank < right_rank {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        if left_rank == right_rank {
            self.rank[left] = left_rank.saturating_add(1);
        }
        left
    }
}
