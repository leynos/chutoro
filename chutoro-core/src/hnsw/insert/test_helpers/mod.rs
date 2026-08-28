//! Test-only helpers for repairing graph connectivity and reciprocity.

mod edge_helpers;

pub(crate) use edge_helpers::add_edge_if_missing;
pub(super) use edge_helpers::assert_no_edge;
#[cfg(test)]
pub(crate) use edge_helpers::{EdgeSymmetry, assert_bidirectional_edge, edge_symmetry};

use super::{
    connectivity::ConnectivityHealer, limits::compute_connection_limit,
    reconciliation::EdgeReconciler, types::UpdateContext,
};
use crate::hnsw::graph::Graph;

#[derive(Debug)]
pub(super) struct TestHelpers<'graph> {
    pub(super) graph: &'graph mut Graph,
}

impl<'graph> TestHelpers<'graph> {
    pub(super) const fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
    }

    #[cfg_attr(
        not(debug_assertions),
        expect(dead_code, reason = "test helper unused in release builds")
    )]
    pub(super) fn heal_reachability(&mut self, max_connections: usize) {
        let Some(entry) = self.graph.entry() else {
            return;
        };

        loop {
            let visited = self.collect_reachable(entry.node);
            let unreachable: Vec<usize> = self
                .graph
                .nodes_iter()
                .map(|(id, _)| id)
                .filter(|&id| !visited.get(id).copied().unwrap_or(false))
                .collect();

            if unreachable.is_empty() {
                break;
            }

            let mut progress = false;
            for node_id in unreachable {
                progress |= self.try_connect_unreachable_node(node_id, &visited, max_connections);
            }

            if !progress {
                break;
            }
        }
    }

    pub(super) fn try_connect_unreachable_node(
        &mut self,
        node_id: usize,
        visited: &[bool],
        max_connections: usize,
    ) -> bool {
        let base_limit = compute_connection_limit(0, max_connections);
        if let Some(origin) = self.first_reachable_with_capacity(visited, base_limit) {
            let ctx = UpdateContext {
                origin,
                level: 0,
                max_connections,
            };
            let mut healer = ConnectivityHealer::new(self.graph);
            if healer.link_new_node(&ctx, node_id) {
                #[cfg(test)]
                self.graph.record_touched_nodes([(origin, 0), (node_id, 0)]);
                return true;
            }
        }

        if let Some(origin) = self.first_reachable(visited) {
            let ctx = UpdateContext {
                origin,
                level: 0,
                max_connections,
            };
            let mut healer = ConnectivityHealer::new(self.graph);
            if healer.link_new_node(&ctx, node_id) {
                #[cfg(test)]
                self.graph.record_touched_nodes([(origin, 0), (node_id, 0)]);
                return true;
            }
        }

        false
    }

    #[cfg_attr(
        not(debug_assertions),
        expect(dead_code, reason = "test helper unused in release builds")
    )]
    #[expect(
        clippy::excessive_nesting,
        reason = "test-only BFS uses simple inline queue"
    )]
    pub(super) fn collect_reachable(&self, entry: usize) -> Vec<bool> {
        let mut visited = vec![false; self.graph.capacity()];
        let mut queue = vec![entry];
        while let Some(next) = queue.pop() {
            if let Some(is_visited) = visited.get_mut(next)
                && !*is_visited
            {
                *is_visited = true;
                if let Some(node_ref) = self.graph.node(next) {
                    queue.extend(node_ref.iter_neighbours().map(|(_, neighbour)| neighbour));
                }
            }
        }
        visited
    }

    #[cfg_attr(
        not(debug_assertions),
        expect(dead_code, reason = "test helper unused in release builds")
    )]
    pub(super) fn first_reachable_with_capacity(
        &self,
        visited: &[bool],
        limit: usize,
    ) -> Option<usize> {
        self.graph
            .nodes_iter()
            .find(|(id, node)| {
                visited.get(*id).copied().unwrap_or(false)
                    && node.level_count() > 0
                    && node.neighbours(0).len() < limit
            })
            .map(|(id, _)| id)
    }

    #[cfg_attr(
        not(debug_assertions),
        expect(dead_code, reason = "test helper unused in release builds")
    )]
    pub(super) fn first_reachable(&self, visited: &[bool]) -> Option<usize> {
        self.graph
            .nodes_iter()
            .map(|(id, _)| id)
            .find(|&id| visited.get(id).copied().unwrap_or(false))
    }

    #[cfg_attr(
        not(debug_assertions),
        expect(dead_code, reason = "test helper unused in release builds")
    )]
    pub(super) fn enforce_bidirectional_all(&mut self, max_connections: usize) {
        for (origin, level, target) in self.collect_edges() {
            let ctx = UpdateContext {
                origin,
                level,
                max_connections,
            };
            self.heal_or_remove_edge(&ctx, target);
        }
    }

    pub(super) fn collect_edges(&self) -> Vec<(usize, usize, usize)> {
        self.graph
            .nodes_iter()
            .flat_map(|(origin, node)| {
                node.iter_neighbours()
                    .map(move |(level, target)| (origin, level, target))
            })
            .collect()
    }

    pub(super) fn heal_or_remove_edge(&mut self, ctx: &UpdateContext, target: usize) {
        if let Some(target_node) = self.graph.node_mut(target)
            && ctx.level < target_node.level_count()
        {
            let limit = compute_connection_limit(ctx.level, ctx.max_connections);
            let Some(neighbours) = target_node.neighbours_mut(ctx.level) else {
                return;
            };
            if neighbours.contains(&ctx.origin) {
                return;
            }

            if neighbours.len() < limit {
                neighbours.push(ctx.origin);
                return;
            }
        }

        let mut reconciler = EdgeReconciler::new(self.graph);
        reconciler.remove_forward_edge_from(ctx, target);
    }

    /// Returns the first edge that breaks reciprocity, or `None` when every
    /// edge is mutual.
    ///
    /// A pure query so callers assert on the outcome at their own call site.
    pub(crate) fn find_reciprocity_violation(
        &self,
        max_connections: usize,
    ) -> Option<ReciprocityViolation> {
        self.collect_edges()
            .into_iter()
            .find_map(|edge| self.edge_reciprocity_violation(edge, max_connections))
    }

    /// Checks a single `(origin, level, target)` edge for reciprocity.
    fn edge_reciprocity_violation(
        &self,
        (origin, level, target): (usize, usize, usize),
        max_connections: usize,
    ) -> Option<ReciprocityViolation> {
        let Some(target_node) = self.graph.node(target) else {
            return Some(ReciprocityViolation::MissingTarget {
                origin,
                target,
                level,
            });
        };

        let target_levels = target_node.level_count();
        if level >= target_levels {
            return Some(ReciprocityViolation::AbsentLevel {
                origin,
                target,
                level,
                target_levels,
            });
        }

        let neighbours = target_node.neighbours(level);
        if neighbours.contains(&origin) {
            return None;
        }

        Some(ReciprocityViolation::OneWay {
            origin,
            target,
            level,
            target_degree: neighbours.len(),
            limit: compute_connection_limit(level, max_connections),
        })
    }

    /// Repairs and validates only edges owned by graph nodes changed by a test mutation.
    pub(super) fn enforce_bidirectional_for_touched(
        &mut self,
        touched: &[(usize, usize)],
        max_connections: usize,
    ) {
        for (origin, level, target) in self.collect_touched_edges(touched) {
            let ctx = UpdateContext {
                origin,
                level,
                max_connections,
            };
            self.heal_or_remove_edge(&ctx, target);
        }

        self.validate_touched_edges_reciprocal(touched, max_connections);
    }

    fn collect_touched_edges(&self, touched: &[(usize, usize)]) -> Vec<(usize, usize, usize)> {
        let mut edges = Vec::new();
        for &(origin, level) in touched {
            let Some(node) = self.graph.node(origin) else {
                continue;
            };
            if level >= node.level_count() {
                continue;
            }
            edges.extend(
                node.neighbours(level)
                    .iter()
                    .copied()
                    .map(|target| (origin, level, target)),
            );
        }
        edges
    }

    fn validate_touched_edges_reciprocal(
        &self,
        touched: &[(usize, usize)],
        max_connections: usize,
    ) {
        for (origin, level, target) in self.collect_touched_edges(touched) {
            let Some(target_node) = self.graph.node(target) else {
                panic!(
                    "enforce_bidirectional_for_touched left edge {origin}->{target} at level {level} to missing node",
                );
            };
            let target_levels = target_node.level_count();
            assert!(
                level < target_levels,
                "enforce_bidirectional_for_touched left edge {origin}->{target} at absent level {level} (target has {target_levels})",
            );

            let neighbours = target_node.neighbours(level);
            let limit = compute_connection_limit(level, max_connections);
            assert!(
                neighbours.contains(&origin),
                "enforce_bidirectional_for_touched left one-way edge {origin}->{target} at level {level}; target degree {} (limit {limit})",
                neighbours.len(),
            );
        }
    }
}

/// Describes why an edge left by [`TestHelpers::enforce_bidirectional_all`]
/// fails the reciprocity invariant.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ReciprocityViolation {
    /// The edge points at a node that is absent from the graph.
    MissingTarget {
        origin: usize,
        target: usize,
        level: usize,
    },
    /// The target node does not expose the edge's level.
    AbsentLevel {
        origin: usize,
        target: usize,
        level: usize,
        target_levels: usize,
    },
    /// The target node does not link back to the origin.
    OneWay {
        origin: usize,
        target: usize,
        level: usize,
        target_degree: usize,
        limit: usize,
    },
}
