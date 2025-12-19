//! Union-find (disjoint set union) utilities used during hierarchy extraction.
//!
//! The hierarchy extractor builds a dendrogram from the mutual-reachability MST
//! by processing edges in non-decreasing order and merging connected
//! components. This module provides the union-find structure used to track
//! component membership.

#[derive(Clone, Debug)]
pub(super) struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
    pub(super) component_node: Vec<usize>,
}

impl DisjointSet {
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            component_node: (0..n).collect(),
        }
    }

    pub(super) fn find(&mut self, mut node: usize) -> usize {
        let mut root = node;
        while self.parent[root] != root {
            root = self.parent[root];
        }

        while self.parent[node] != node {
            let parent = self.parent[node];
            self.parent[node] = root;
            node = parent;
        }

        root
    }

    pub(super) fn union(&mut self, left: usize, right: usize) -> usize {
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
