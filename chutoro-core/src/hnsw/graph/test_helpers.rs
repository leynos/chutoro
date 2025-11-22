use std::collections::HashSet;

use super::core::Graph;
use crate::hnsw::{
    error::HnswError,
    params::{self, HnswParams},
    types::EntryPoint,
};

impl Graph {
    pub(crate) fn set_params(&mut self, params: &HnswParams) {
        self.params = params.clone();
    }

    pub(crate) fn delete_node(&mut self, node: usize) -> Result<bool, HnswError> {
        if node >= self.nodes.len() {
            return Err(HnswError::InvalidParameters {
                reason: format!("node {node} exceeds graph capacity {}", self.nodes.len()),
            });
        }
        let slot = self.nodes.get_mut(node).expect("bounds checked above");
        let Some(removed) = slot.take() else {
            return Ok(false);
        };

        let removed_neighbours: Vec<Vec<usize>> = (0..removed.level_count())
            .map(|level| removed.neighbours(level).to_vec())
            .collect();

        for maybe_node in self.nodes.iter_mut().flatten() {
            let levels = maybe_node.level_count();
            for level in 0..levels {
                let neighbours = maybe_node.neighbours_mut(level);
                neighbours.retain(|&target| target != node);
            }
        }

        for (level, neighbours) in removed_neighbours.into_iter().enumerate() {
            self.reconnect_neighbours(level, neighbours);
        }

        if self.entry.map(|entry| entry.node) == Some(node) {
            self.entry = self.recompute_entry_point();
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
}
