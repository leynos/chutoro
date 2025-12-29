//! Test-only helpers for mutating the CPU HNSW graph.
//!
//! Provides deletion and reconnection utilities used exclusively by the
//! property-based mutation harness. The helpers trade performance for
//! correctness and observability, ensuring reachability is preserved or the
//! mutation is rolled back when it would fragment the graph.

use std::collections::{HashSet, VecDeque};

use super::core::Graph;
use crate::hnsw::{
    error::HnswError,
    node::Node,
    params::{self, HnswParams},
    types::EntryPoint,
};

impl Graph {
    pub(crate) fn set_params(&mut self, params: &HnswParams) {
        self.params = params.clone();
    }

    pub(crate) fn delete_node(&mut self, node: usize) -> Result<bool, HnswError> {
        self.validate_delete_target(node)?;
        let Some(existing) = self.nodes.get(node).and_then(Option::as_ref) else {
            return Ok(false);
        };

        let snapshot_nodes = self.nodes.clone();
        let snapshot_entry = self.entry;
        let removed_neighbours = collect_neighbour_layers(existing);

        self.nodes
            .get_mut(node)
            .and_then(Option::take)
            .expect("node presence checked above");

        self.strip_references_to(node);
        self.reconnect_layers(removed_neighbours);

        if self.entry.map(|entry| entry.node) == Some(node) {
            self.entry = self.recompute_entry_point();
        }

        if let Err(err) = self.ensure_reachability() {
            self.nodes = snapshot_nodes;
            self.entry = snapshot_entry;
            return Err(err);
        }

        Ok(true)
    }

    pub(super) fn recompute_entry_point(&self) -> Option<EntryPoint> {
        self.nodes_iter()
            .max_by_key(|(id, node)| (node.level_count(), std::cmp::Reverse(*id)))
            .map(|(id, node)| EntryPoint {
                node: id,
                level: node.level_count().saturating_sub(1),
            })
    }

    pub(super) fn reconnect_neighbours(&mut self, level: usize, neighbours: Vec<usize>) {
        let unique = self.validate_and_dedupe_neighbours(neighbours);

        if unique.len() < 2 {
            return;
        }

        self.connect_neighbour_pairs(&unique, level);
    }

    /// Validates and deduplicates a list of neighbours, keeping only valid nodes.
    pub(super) fn validate_and_dedupe_neighbours(&self, neighbours: Vec<usize>) -> Vec<usize> {
        let mut unique = Vec::new();
        let mut seen = HashSet::new();

        for neighbour in neighbours {
            if self.is_valid_unique_neighbour(neighbour, &mut seen) {
                unique.push(neighbour);
            }
        }

        unique
    }

    /// Checks if a neighbour is both unique and points to a valid node.
    pub(super) fn is_valid_unique_neighbour(
        &self,
        neighbour: usize,
        seen: &mut HashSet<usize>,
    ) -> bool {
        if !seen.insert(neighbour) {
            return false;
        }

        self.nodes.get(neighbour).and_then(Option::as_ref).is_some()
    }

    /// Connects consecutive pairs of neighbours bidirectionally at the given level.
    pub(super) fn connect_neighbour_pairs(&mut self, unique: &[usize], level: usize) {
        for pair in unique.windows(2) {
            if let [origin, target] = pair {
                self.try_add_bidirectional_edge(*origin, *target, level);
            }
        }
    }

    pub(super) fn try_add_bidirectional_edge(
        &mut self,
        origin: usize,
        target: usize,
        level: usize,
    ) {
        let added_forward = self.try_add_edge(origin, target, level);
        let added_reverse = self.try_add_edge(target, origin, level);

        if added_forward && !added_reverse {
            self.remove_edge(origin, target, level);
        }
        if added_reverse && !added_forward {
            self.remove_edge(target, origin, level);
        }
    }

    pub(super) fn try_add_edge(&mut self, origin: usize, target: usize, level: usize) -> bool {
        let limit = params::connection_limit_for_level(level, self.params.max_connections());
        let Some(node) = self.nodes.get_mut(origin).and_then(Option::as_mut) else {
            return false;
        };
        if level >= node.level_count() {
            return false;
        }

        let neighbours = node.neighbours_mut(level);
        if neighbours.contains(&target) {
            return true;
        }

        if neighbours.len() < limit {
            neighbours.push(target);
            return true;
        }

        false
    }

    pub(super) fn remove_edge(&mut self, origin: usize, target: usize, level: usize) {
        let Some(node) = self.nodes.get_mut(origin).and_then(Option::as_mut) else {
            return;
        };
        if level >= node.level_count() {
            return;
        }

        let neighbours = node.neighbours_mut(level);
        if let Some(pos) = neighbours.iter().position(|&candidate| candidate == target) {
            neighbours.remove(pos);
        }
    }

    fn validate_delete_target(&self, node: usize) -> Result<(), HnswError> {
        if node >= self.nodes.len() {
            return Err(HnswError::InvalidParameters {
                reason: format!("node {node} exceeds graph capacity {}", self.nodes.len()),
            });
        }
        Ok(())
    }

    fn strip_references_to(&mut self, node: usize) {
        for maybe_node in self.nodes.iter_mut().flatten() {
            let levels = maybe_node.level_count();
            for level in 0..levels {
                let neighbours = maybe_node.neighbours_mut(level);
                neighbours.retain(|&target| target != node);
            }
        }
    }

    fn reconnect_layers(&mut self, removed_neighbours: Vec<Vec<usize>>) {
        for (level, neighbours) in removed_neighbours.into_iter().enumerate() {
            self.reconnect_neighbours(level, neighbours);
        }
    }

    fn ensure_reachability(&self) -> Result<(), HnswError> {
        if self.nodes_iter().next().is_none() {
            return Ok(());
        }

        let entry = self
            .entry
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: "entry point missing after delete".into(),
            })?;

        let mut state = ReachabilityState::new(self.nodes.len(), entry.node);

        while let Some(node_id) = state.queue.pop_front() {
            let node = self
                .nodes
                .get(node_id)
                .and_then(Option::as_ref)
                .ok_or_else(|| HnswError::GraphInvariantViolation {
                    message: format!("node {node_id} missing during reachability walk"),
                })?;

            for (_, target) in node.iter_neighbours() {
                self.enqueue_neighbour(node_id, target, &mut state)?;
            }
        }

        if let Some(unreachable) = self
            .nodes_iter()
            .map(|(id, _)| id)
            .find(|&id| !state.visited[id])
        {
            return Err(HnswError::GraphInvariantViolation {
                message: format!(
                    "delete would disconnect node {unreachable} from entry {}",
                    entry.node
                ),
            });
        }

        Ok(())
    }

    fn enqueue_neighbour(
        &self,
        origin: usize,
        target: usize,
        state: &mut ReachabilityState,
    ) -> Result<(), HnswError> {
        if target >= self.nodes.len() {
            return Err(HnswError::GraphInvariantViolation {
                message: format!("node {origin} references out-of-bounds neighbour {target}"),
            });
        }
        if self.nodes.get(target).and_then(Option::as_ref).is_none() {
            return Err(HnswError::GraphInvariantViolation {
                message: format!("node {origin} references missing neighbour {target}"),
            });
        }
        if state.visited[target] {
            return Ok(());
        }

        state.visit(target);
        Ok(())
    }
}

struct ReachabilityState {
    visited: Vec<bool>,
    queue: VecDeque<usize>,
}

impl ReachabilityState {
    fn new(capacity: usize, entry: usize) -> Self {
        let mut visited = vec![false; capacity];
        let mut queue = VecDeque::new();
        visited[entry] = true;
        queue.push_back(entry);
        Self { visited, queue }
    }

    fn visit(&mut self, node: usize) {
        if self.visited[node] {
            return;
        }
        self.visited[node] = true;
        self.queue.push_back(node);
    }
}

fn collect_neighbour_layers(removed: &Node) -> Vec<Vec<usize>> {
    (0..removed.level_count())
        .map(|level| removed.neighbours(level).to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{HnswError, graph::NodeContext};
    use rstest::rstest;

    #[rstest]
    fn delete_node_reconnects_neighbours_and_preserves_reachability() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 3);
        graph
            .insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })
            .expect("insert entry");
        graph
            .attach_node(NodeContext {
                node: 1,
                level: 0,
                sequence: 1,
            })
            .expect("attach first neighbour");
        graph
            .attach_node(NodeContext {
                node: 2,
                level: 0,
                sequence: 2,
            })
            .expect("attach second neighbour");
        graph.try_add_bidirectional_edge(0, 1, 0);
        graph.try_add_bidirectional_edge(1, 2, 0);

        let deleted = graph.delete_node(1).expect("delete must succeed");

        assert!(deleted, "node should be removed");
        assert!(
            graph.node(1).is_none(),
            "slot 1 must be cleared after deletion"
        );
        let node0 = graph.node(0).expect("node 0 must remain");
        assert_eq!(node0.neighbours(0), &[2], "node 0 must connect to node 2");
        let node2 = graph.node(2).expect("node 2 must remain");
        assert_eq!(node2.neighbours(0), &[0], "node 2 must connect to node 0");
        assert_eq!(graph.entry().map(|entry| entry.node), Some(0));
    }

    #[rstest]
    fn delete_node_returns_ok_false_for_missing_node() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 3);

        graph
            .insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })
            .expect("insert entry");

        let first_delete = graph.delete_node(0).expect("delete existing node");
        assert!(
            first_delete,
            "expected Ok(true) when deleting an existing node"
        );

        let second_delete = graph.delete_node(0);
        assert!(
            matches!(second_delete, Ok(false)),
            "expected Ok(false) when deleting a missing node, got {second_delete:?}",
        );

        let third_delete = graph.delete_node(0);
        assert!(
            matches!(third_delete, Ok(false)),
            "expected Ok(false) on repeated deletes of a missing node, got {third_delete:?}",
        );
    }

    #[rstest]
    fn delete_node_returns_invalid_parameters_for_out_of_bounds_index() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 3);

        let result = graph.delete_node(5);

        match result {
            Err(HnswError::InvalidParameters { .. }) => {}
            other => panic!(
                "expected Err(HnswError::InvalidParameters {{ .. }}), got {other:?}",
            ),
        }
    }

    #[rstest]
    fn delete_node_reverts_when_it_would_disconnect_graph() {
        let params = HnswParams::new(1, 1).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 5);
        graph
            .insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })
            .expect("insert entry");
        for (node, sequence) in [(1_usize, 1_u64), (2, 2), (3, 3), (4, 4)] {
            graph
                .attach_node(NodeContext {
                    node,
                    level: 0,
                    sequence,
                })
                .expect("attach node");
        }
        graph.try_add_bidirectional_edge(0, 1, 0);
        graph.try_add_bidirectional_edge(0, 2, 0);
        graph.try_add_bidirectional_edge(0, 3, 0);
        graph.try_add_bidirectional_edge(1, 4, 0);
        graph.try_add_bidirectional_edge(2, 4, 0);

        let result = graph.delete_node(0);

        let err = result.expect_err("delete must fail to preserve reachability");
        match err {
            HnswError::GraphInvariantViolation { .. } => {}
            other => panic!("expected GraphInvariantViolation, got {other:?}"),
        }
        assert!(
            graph.node(0).is_some(),
            "failed deletion must restore the removed node"
        );
        let node1 = graph.node(1).expect("node 1 must remain");
        assert!(
            node1.neighbours(0).contains(&0),
            "node 1 should retain its link to the entry"
        );
        assert_eq!(
            graph.entry().map(|entry| entry.node),
            Some(0),
            "entry point must roll back on failure"
        );
    }
}
