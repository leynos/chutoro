//! Condensation utilities for single-linkage dendrograms.
//!
//! The condensed tree follows the HDBSCAN procedure: clusters below
//! `min_cluster_size` are treated as noise, and a parent cluster only "splits"
//! when both children satisfy the minimum size.

use super::{CondensedCluster, CondensedEvent, SingleLinkageForest};

pub(super) struct CondenseBuilder<'a> {
    forest: &'a SingleLinkageForest,
    min_cluster_size: usize,
    clusters: &'a mut Vec<CondensedCluster>,
}

impl<'a> CondenseBuilder<'a> {
    pub(super) fn new(
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

    pub(super) fn condense_cluster(&mut self, node_id: usize, cluster_id: usize) {
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
