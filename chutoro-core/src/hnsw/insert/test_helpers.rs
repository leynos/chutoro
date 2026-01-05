//! Test-only helpers for repairing graph connectivity and reciprocity.

use super::{
    connectivity::ConnectivityHealer, limits::compute_connection_limit,
    reconciliation::EdgeReconciler, types::UpdateContext,
};
use crate::hnsw::graph::Graph;

pub(crate) fn add_edge_if_missing(graph: &mut Graph, origin: usize, target: usize, level: usize) {
    let msg = format!("node {origin} should exist");
    let node = graph.node_mut(origin).expect(&msg);
    let neighbours = node.neighbours_mut(level);
    if !neighbours.contains(&target) {
        neighbours.push(target);
    }
}

pub(super) fn assert_no_edge(graph: &Graph, origin: usize, target: usize, level: usize) {
    if let Some(node) = graph.node(origin) {
        if level < node.level_count() {
            assert!(
                !node.neighbours(level).contains(&target),
                "unexpected edge {origin}->{target} at level {level}",
            );
        }
    }
}

#[derive(Debug)]
pub(super) struct TestHelpers<'graph> {
    pub(super) graph: &'graph mut Graph,
}

impl<'graph> TestHelpers<'graph> {
    pub(super) fn new(graph: &'graph mut Graph) -> Self {
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
            if next < visited.len() && !visited[next] {
                visited[next] = true;
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

        self.validate_all_edges_reciprocal(max_connections);
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

    #[expect(
        clippy::excessive_nesting,
        reason = "Test helper keeps explicit fallbacks for clarity"
    )]
    pub(super) fn heal_or_remove_edge(&mut self, ctx: &UpdateContext, target: usize) {
        if let Some(target_node) = self.graph.node_mut(target) {
            if ctx.level < target_node.level_count() {
                let limit = compute_connection_limit(ctx.level, ctx.max_connections);
                let neighbours = target_node.neighbours_mut(ctx.level);
                if neighbours.contains(&ctx.origin) {
                    return;
                }

                if neighbours.len() < limit {
                    neighbours.push(ctx.origin);
                    return;
                }
            }
        }

        let mut reconciler = EdgeReconciler::new(self.graph);
        reconciler.remove_forward_edge_from(ctx, target);
    }

    #[expect(
        clippy::excessive_nesting,
        reason = "test-only reciprocal validation keeps explicit panic messages"
    )]
    pub(super) fn validate_all_edges_reciprocal(&self, max_connections: usize) {
        for (origin, node) in self.graph.nodes_iter() {
            for (level, target) in node.iter_neighbours() {
                let target_node = match self.graph.node(target) {
                    Some(node) => node,
                    None => {
                        panic!(
                            "enforce_bidirectional_all left edge {origin}->{target} at level {level} to missing node",
                        );
                    }
                };

                let target_levels = target_node.level_count();
                assert!(
                    level < target_levels,
                    "enforce_bidirectional_all left edge {origin}->{target} at absent level {level} (target has {target_levels})",
                );

                let neighbours = target_node.neighbours(level);
                let limit = compute_connection_limit(level, max_connections);
                assert!(
                    neighbours.contains(&origin),
                    "enforce_bidirectional_all left one-way edge {origin}->{target} at level {level}; target degree {} (limit {limit})",
                    neighbours.len(),
                );
            }
        }
    }
}
