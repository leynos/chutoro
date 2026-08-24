//! Condensation utilities for single-linkage dendrograms.
//!
//! The condensed tree follows the HDBSCAN procedure: clusters below
//! `min_cluster_size` are treated as noise, and a parent cluster only "splits"
//! when both children satisfy the minimum size.

use core::ops::{AddAssign, Div, Mul, Sub};

use num_traits::cast;

use super::{CondensedCluster, CondensedEvent, HierarchyError, SingleLinkageForest};

pub(super) struct CondenseBuilder<'a> {
    forest: &'a SingleLinkageForest,
    min_cluster_size: usize,
    clusters: &'a mut Vec<CondensedCluster>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SplitCase {
    BothBig,
    LeftBigOnly,
    RightBigOnly,
    BothSmall,
}

impl SplitCase {
    const fn from_flags(left_big: bool, right_big: bool) -> Self {
        match (left_big, right_big) {
            (true, true) => Self::BothBig,
            (true, false) => Self::LeftBigOnly,
            (false, true) => Self::RightBigOnly,
            (false, false) => Self::BothSmall,
        }
    }
}

impl<'a> CondenseBuilder<'a> {
    pub(super) const fn new(
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

    pub(super) fn condense_cluster(
        &mut self,
        node_id: usize,
        cluster_id: usize,
    ) -> Result<(), HierarchyError> {
        let node = self
            .forest
            .nodes
            .get(node_id)
            .ok_or(HierarchyError::InvalidForestReference { node_id })?;
        let Some((left, right)) = node.left.zip(node.right) else {
            if let Some(point) = node.point {
                record_point_event(self.clusters, cluster_id, point, f32::INFINITY)?;
            }
            return Ok(());
        };

        let lambda = weight_to_lambda(node.weight);
        let left_size = self
            .forest
            .nodes
            .get(left)
            .ok_or(HierarchyError::InvalidForestReference { node_id: left })?
            .size;
        let right_size = self
            .forest
            .nodes
            .get(right)
            .ok_or(HierarchyError::InvalidForestReference { node_id: right })?
            .size;
        let left_big = left_size >= self.min_cluster_size;
        let right_big = right_size >= self.min_cluster_size;

        match SplitCase::from_flags(left_big, right_big) {
            SplitCase::BothBig => {
                let left_cluster = self.create_child_cluster(cluster_id, lambda, left_size)?;
                let right_cluster = self.create_child_cluster(cluster_id, lambda, right_size)?;
                self.condense_cluster(left, left_cluster)?;
                self.condense_cluster(right, right_cluster)?;
            }
            SplitCase::LeftBigOnly => {
                self.emit_pruned_points(right, cluster_id, lambda)?;
                self.condense_cluster(left, cluster_id)?;
            }
            SplitCase::RightBigOnly => {
                self.emit_pruned_points(left, cluster_id, lambda)?;
                self.condense_cluster(right, cluster_id)?;
            }
            SplitCase::BothSmall => {
                self.emit_pruned_points(left, cluster_id, lambda)?;
                self.emit_pruned_points(right, cluster_id, lambda)?;
            }
        }

        Ok(())
    }

    fn create_child_cluster(
        &mut self,
        parent: usize,
        lambda: f32,
        size: usize,
    ) -> Result<usize, HierarchyError> {
        if self.clusters.get(parent).is_none() {
            return Err(HierarchyError::InvalidClusterReference { cluster_id: parent });
        }

        let child_id = self.clusters.len();
        self.clusters
            .push(CondensedCluster::new(Some(parent), lambda));
        let parent_cluster = self
            .clusters
            .get_mut(parent)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id: parent })?;
        parent_cluster.children.push(child_id);
        parent_cluster.events.push(CondensedEvent::ChildCluster {
            cluster: child_id,
            lambda,
            size,
        });
        record_stability_increment(parent_cluster, lambda, narrow_size_to_f32(size));
        Ok(child_id)
    }

    fn emit_pruned_points(
        &mut self,
        node_id: usize,
        cluster_id: usize,
        lambda: f32,
    ) -> Result<(), HierarchyError> {
        let mut stack = vec![node_id];
        while let Some(current) = stack.pop() {
            let node = self
                .forest
                .nodes
                .get(current)
                .ok_or(HierarchyError::InvalidForestReference { node_id: current })?;
            if let Some(point) = node.point {
                record_point_event(self.clusters, cluster_id, point, lambda)?;
                continue;
            }
            if let Some(left) = node.left {
                stack.push(left);
            }
            if let Some(right) = node.right {
                stack.push(right);
            }
        }

        Ok(())
    }
}

fn record_point_event(
    clusters: &mut [CondensedCluster],
    cluster_id: usize,
    point: usize,
    lambda: f32,
) -> Result<(), HierarchyError> {
    let cluster = clusters
        .get_mut(cluster_id)
        .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?;
    cluster.events.push(CondensedEvent::Point {
        index: point,
        lambda,
    });
    record_stability_increment(cluster, lambda, 1.0);
    Ok(())
}

fn record_stability_increment(cluster: &mut CondensedCluster, lambda: f32, size: f32) {
    let increment = lambda.sub(cluster.birth_lambda).mul(size);
    cluster.stability.add_assign(increment);
}

fn weight_to_lambda(weight: f32) -> f32 {
    if weight == 0.0 {
        f32::INFINITY
    } else {
        1.0_f32.div(weight)
    }
}

fn narrow_size_to_f32(size: usize) -> f32 {
    cast(size).unwrap_or(f32::NAN)
}
