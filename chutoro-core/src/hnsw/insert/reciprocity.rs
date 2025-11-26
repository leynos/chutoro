//! Ensures reciprocal edges for nodes touched during insertion.
//!
//! Reciprocity enforcement cleans up one-way edges that can emerge after
//! trimming and fallback linking. It also provides a workspace to adjust the
//! new node's neighbour lists based on which connections were successfully
//! reciprocated.

use std::collections::HashSet;

use crate::hnsw::graph::Graph;

use super::{
    limits::compute_connection_limit,
    reconciliation::EdgeReconciler,
    types::{FinalisedUpdate, UpdateContext},
};

#[derive(Debug)]
pub(super) struct ReciprocityEnforcer<'graph> {
    pub(super) graph: &'graph mut Graph,
}

impl<'graph> ReciprocityEnforcer<'graph> {
    pub(super) fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
    }

    pub(super) fn ensure_reciprocity_for_touched(
        &mut self,
        touched: &[(usize, usize)],
        max_connections: usize,
    ) {
        let mut seen = HashSet::new();
        for &(origin, level) in touched {
            if !seen.insert((origin, level)) {
                continue;
            }
            self.ensure_reciprocity_for_node_level(origin, level, max_connections);
        }
    }

    pub(super) fn ensure_reciprocity_for_node_level(
        &mut self,
        origin: usize,
        level: usize,
        max_connections: usize,
    ) {
        let neighbours_snapshot = {
            let graph_ref = &*self.graph;
            graph_ref
                .node(origin)
                .filter(|node| node.level_count() > level)
                .map(|node| node.neighbours(level).to_vec())
        };

        let Some(neighbours_snapshot) = neighbours_snapshot else {
            return;
        };

        let ctx = UpdateContext {
            origin,
            level,
            max_connections,
        };

        let mut reconciler = EdgeReconciler::new(self.graph);
        for target in neighbours_snapshot {
            if reconciler.ensure_reverse_edge(&ctx, target) {
                continue;
            }
            reconciler.remove_forward_edge_from(&ctx, target);
        }
    }
}

pub(super) struct ReciprocityWorkspace<'a> {
    pub(super) filtered: &'a mut [Vec<usize>],
    pub(super) original: &'a [Vec<usize>],
    pub(super) final_updates: &'a mut [FinalisedUpdate],
    pub(super) new_node: usize,
    pub(super) max_connections: usize,
}

impl<'a> ReciprocityWorkspace<'a> {
    pub(super) fn apply(self) {
        let ReciprocityWorkspace {
            filtered,
            original,
            final_updates,
            new_node,
            max_connections,
        } = self;

        let mut selector = FallbackSelector {
            original,
            final_updates,
            new_node,
            max_connections,
        };

        for (level, neighbours) in filtered.iter_mut().enumerate() {
            let reciprocated = selector.reciprocated(level);
            neighbours.retain(|candidate| reciprocated.contains(candidate));

            if !neighbours.is_empty() {
                continue;
            }

            if let Some(candidate) = selector.select(level) {
                neighbours.push(candidate);
            }
        }
    }
}

struct FallbackSelector<'a> {
    original: &'a [Vec<usize>],
    final_updates: &'a mut [FinalisedUpdate],
    new_node: usize,
    max_connections: usize,
}

impl<'a> FallbackSelector<'a> {
    fn reciprocated(&self, level: usize) -> HashSet<usize> {
        self.final_updates
            .iter()
            .filter_map(|(update, neighbours)| {
                (update.ctx.level == level && neighbours.contains(&self.new_node))
                    .then_some(update.node)
            })
            .collect()
    }

    fn select(&mut self, level: usize) -> Option<usize> {
        let fallback_candidates = self.original.get(level).map(Vec::as_slice).unwrap_or(&[]);
        let limit = compute_connection_limit(level, self.max_connections);

        for &candidate in fallback_candidates {
            let Some((_, neighbour_list)) = self
                .final_updates
                .iter_mut()
                .find(|(update, _)| update.node == candidate && update.ctx.level == level)
            else {
                continue;
            };

            if neighbour_list.contains(&self.new_node) {
                return Some(candidate);
            }
            if neighbour_list.len() < limit {
                neighbour_list.push(self.new_node);
                return Some(candidate);
            }
            if !neighbour_list.is_empty() {
                neighbour_list.pop();
                neighbour_list.push(self.new_node);
                return Some(candidate);
            }
        }

        None
    }
}
