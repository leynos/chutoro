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

/// Captures one dendrogram split for the internal condensation transition.
///
/// This is local to [`CondenseBuilder`]: it groups the two child identifiers,
/// their sizes, and the split lambda without widening the hierarchy API.
struct BranchSplit {
    left: usize,
    right: usize,
    lambda: f32,
    left_size: usize,
    right_size: usize,
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
        let Some((left, right, lambda)) = self.branch_details(node_id)? else {
            return self.record_leaf(node_id, cluster_id);
        };

        let left_size = self.node_size(left)?;
        let right_size = self.node_size(right)?;
        let left_big = left_size >= self.min_cluster_size;
        let right_big = right_size >= self.min_cluster_size;

        let branch_split = BranchSplit {
            left,
            right,
            lambda,
            left_size,
            right_size,
        };
        self.apply_split_case(
            SplitCase::from_flags(left_big, right_big),
            cluster_id,
            &branch_split,
        )
    }

    fn branch_details(
        &self,
        node_id: usize,
    ) -> Result<Option<(usize, usize, f32)>, HierarchyError> {
        let node = self
            .forest
            .nodes
            .get(node_id)
            .ok_or(HierarchyError::InvalidForestReference { node_id })?;
        Ok(node
            .left
            .zip(node.right)
            .map(|(left, right)| (left, right, weight_to_lambda(node.weight))))
    }

    fn node_size(&self, node_id: usize) -> Result<usize, HierarchyError> {
        self.forest
            .nodes
            .get(node_id)
            .map(|node| node.size)
            .ok_or(HierarchyError::InvalidForestReference { node_id })
    }

    fn record_leaf(&mut self, node_id: usize, cluster_id: usize) -> Result<(), HierarchyError> {
        let node = self
            .forest
            .nodes
            .get(node_id)
            .ok_or(HierarchyError::InvalidForestReference { node_id })?;
        if let Some(point) = node.point {
            record_point_event(self.clusters, cluster_id, point, f32::INFINITY)?;
        }
        Ok(())
    }

    fn apply_split_case(
        &mut self,
        split_case: SplitCase,
        cluster_id: usize,
        branch_split: &BranchSplit,
    ) -> Result<(), HierarchyError> {
        match split_case {
            SplitCase::BothBig => self.split_both_big(cluster_id, branch_split),
            SplitCase::LeftBigOnly => self.split_left_big_only(cluster_id, branch_split),
            SplitCase::RightBigOnly => self.split_right_big_only(cluster_id, branch_split),
            SplitCase::BothSmall => self.split_both_small(cluster_id, branch_split),
        }
    }

    fn split_both_big(
        &mut self,
        cluster_id: usize,
        branch_split: &BranchSplit,
    ) -> Result<(), HierarchyError> {
        let left_cluster =
            self.create_child_cluster(cluster_id, branch_split.lambda, branch_split.left_size)?;
        let right_cluster =
            self.create_child_cluster(cluster_id, branch_split.lambda, branch_split.right_size)?;
        self.condense_cluster(branch_split.left, left_cluster)?;
        self.condense_cluster(branch_split.right, right_cluster)
    }

    fn split_left_big_only(
        &mut self,
        cluster_id: usize,
        branch_split: &BranchSplit,
    ) -> Result<(), HierarchyError> {
        self.emit_pruned_points(branch_split.right, cluster_id, branch_split.lambda)?;
        self.condense_cluster(branch_split.left, cluster_id)
    }

    fn split_right_big_only(
        &mut self,
        cluster_id: usize,
        branch_split: &BranchSplit,
    ) -> Result<(), HierarchyError> {
        self.emit_pruned_points(branch_split.left, cluster_id, branch_split.lambda)?;
        self.condense_cluster(branch_split.right, cluster_id)
    }

    fn split_both_small(
        &mut self,
        cluster_id: usize,
        branch_split: &BranchSplit,
    ) -> Result<(), HierarchyError> {
        self.emit_pruned_points(branch_split.left, cluster_id, branch_split.lambda)?;
        self.emit_pruned_points(branch_split.right, cluster_id, branch_split.lambda)
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
            let children = self.prune_node(current, cluster_id, lambda)?;
            stack.extend(children.into_iter().flatten());
        }

        Ok(())
    }

    fn prune_node(
        &mut self,
        node_id: usize,
        cluster_id: usize,
        lambda: f32,
    ) -> Result<[Option<usize>; 2], HierarchyError> {
        let node = self
            .forest
            .nodes
            .get(node_id)
            .ok_or(HierarchyError::InvalidForestReference { node_id })?;
        if let Some(point) = node.point {
            record_point_event(self.clusters, cluster_id, point, lambda)?;
            return Ok([None, None]);
        }
        Ok([node.left, node.right])
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
