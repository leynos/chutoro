//! Node storage for the CPU HNSW graph.
//!
//! Maintains per-level neighbour lists and provides accessors used during
//! search, insertion, and trimming.
/// Per-node adjacency lists and deterministic insertion metadata.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    /// Mutable adjacency list for every initialized graph level.
    neighbours: Vec<Vec<usize>>,
    /// Deterministic insertion order for equal-distance tie-breaking.
    sequence: u64,
}

impl Node {
    /// Create a node with empty adjacency lists through `level`.
    pub(crate) fn new(level: usize, sequence: u64) -> Self {
        let mut neighbours = Vec::with_capacity(level + 1);
        neighbours.resize_with(level + 1, Vec::new);
        Self {
            neighbours,
            sequence,
        }
    }

    /// Return neighbours at `level`, or an empty slice when unavailable.
    pub(crate) fn neighbours(&self, level: usize) -> &[usize] {
        self.neighbours.get(level).map_or(&[], Vec::as_slice)
    }

    /// Return mutable neighbours at `level` when that level is initialized.
    pub(crate) fn neighbours_mut(&mut self, level: usize) -> Option<&mut Vec<usize>> {
        self.neighbours.get_mut(level)
    }

    /// Return this node's deterministic insertion sequence.
    pub(crate) const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Returns the number of levels initialized for this node.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use crate::hnsw::{graph::{Graph, NodeContext}, params::HnswParams};
    ///
    /// let params = HnswParams::new(4, 8).expect("params must be valid");
    /// let mut graph = Graph::with_capacity(params, 1);
    /// graph.insert_first(NodeContext { node: 0, level: 1, sequence: 42 }).expect("insert node");
    /// let node = graph.node(0).expect("node 0 must exist");
    /// assert_eq!(node.level_count(), 2);
    /// ```
    #[must_use]
    pub(crate) const fn level_count(&self) -> usize {
        self.neighbours.len()
    }

    /// Iterates over every neighbour across all layers, yielding `(level, id)`.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use crate::hnsw::{graph::{Graph, NodeContext}, params::HnswParams};
    ///
    /// let params = HnswParams::new(4, 8).expect("params");
    /// let mut graph = Graph::with_capacity(params, 2);
    /// graph
    ///     .insert_first(NodeContext { node: 0, level: 0, sequence: 0 })
    ///     .expect("insert first node");
    /// graph
    ///     .attach_node(NodeContext { node: 1, level: 0, sequence: 1 })
    ///     .expect("attach second node");
    /// graph
    ///     .node_mut(0)
    ///     .expect("node 0")
    ///     .neighbours_mut(0)
    ///     .expect("node 0 must expose level 0")
    ///     .push(1);
    /// let neighbours: Vec<_> = graph
    ///     .node(0)
    ///     .expect("node 0")
    ///     .iter_neighbours()
    ///     .collect();
    /// assert_eq!(neighbours, vec![(0, 1)]);
    /// ```
    pub(crate) fn iter_neighbours(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.neighbours
            .iter()
            .enumerate()
            .flat_map(|(level, ids)| ids.iter().copied().map(move |target| (level, target)))
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for node adjacency accessors.

    use super::Node;

    #[test]
    fn unavailable_level_cannot_be_mutated() {
        let mut node = Node::new(0, 0);

        assert!(node.neighbours_mut(1).is_none());
        assert!(node.neighbours(1).is_empty());
    }
}
