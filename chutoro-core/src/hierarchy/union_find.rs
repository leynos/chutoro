//! Union-find (disjoint set union) utilities used during hierarchy extraction.
//!
//! The hierarchy extractor builds a dendrogram from the mutual-reachability MST
//! by processing edges in non-decreasing order and merging connected
//! components. This module provides the union-find structure used to track
//! component membership.

use super::single_linkage::HierarchyError;

/// Track connectivity and linkage-node ownership during forest construction.
#[derive(Clone, Debug)]
pub(super) struct DisjointSet {
    /// Parent pointer for each union-find item.
    parent: Vec<usize>,
    /// Rank used to balance union operations.
    rank: Vec<u8>,
    /// Linkage-node identifier for each component root.
    pub(super) component_node: Vec<usize>,
}

impl DisjointSet {
    /// Create disjoint singleton components for `n` leaf nodes.
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            component_node: (0..n).collect(),
        }
    }

    /// Read a parent pointer or report an invalid forest reference.
    fn parent_at(&self, node: usize) -> Result<usize, HierarchyError> {
        self.parent
            .get(node)
            .copied()
            .ok_or(HierarchyError::InvalidForestReference { node_id: node })
    }

    /// Replace a parent pointer or report an invalid forest reference.
    fn set_parent(&mut self, node: usize, parent: usize) -> Result<(), HierarchyError> {
        *self
            .parent
            .get_mut(node)
            .ok_or(HierarchyError::InvalidForestReference { node_id: node })? = parent;
        Ok(())
    }

    /// Read a component rank or report an invalid forest reference.
    fn rank_at(&self, node: usize) -> Result<u8, HierarchyError> {
        self.rank
            .get(node)
            .copied()
            .ok_or(HierarchyError::InvalidForestReference { node_id: node })
    }

    /// Replace a component rank or report an invalid forest reference.
    fn set_rank(&mut self, node: usize, rank: u8) -> Result<(), HierarchyError> {
        *self
            .rank
            .get_mut(node)
            .ok_or(HierarchyError::InvalidForestReference { node_id: node })? = rank;
        Ok(())
    }

    /// Find a component root while compressing the traversed path.
    pub(super) fn find(&mut self, mut node: usize) -> Result<usize, HierarchyError> {
        let mut root = node;
        while self.parent_at(root)? != root {
            root = self.parent_at(root)?;
        }

        while self.parent_at(node)? != node {
            let parent = self.parent_at(node)?;
            self.set_parent(node, root)?;
            node = parent;
        }

        Ok(root)
    }

    /// Union two components and return their resulting root.
    pub(super) fn union(&mut self, left: usize, right: usize) -> Result<usize, HierarchyError> {
        let mut left_root = self.find(left)?;
        let mut right_root = self.find(right)?;
        if left_root == right_root {
            return Ok(left_root);
        }
        let left_rank = self.rank_at(left_root)?;
        let right_rank = self.rank_at(right_root)?;
        if left_rank < right_rank {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.set_parent(right_root, left_root)?;
        if left_rank == right_rank {
            self.set_rank(left_root, left_rank.saturating_add(1))?;
        }
        Ok(left_root)
    }
}
