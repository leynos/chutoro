//! Test-only helpers for repairing graph connectivity and reciprocity.

// `cargo kani` sets `cfg(kani)` but not `cfg(test)`, so anything reachable
// only from tests must be gated or it becomes dead code under `-D warnings`.
#[cfg(test)]
use super::{
    connectivity::ConnectivityHealer, limits::compute_connection_limit,
    reconciliation::EdgeReconciler, types::UpdateContext,
};
use crate::hnsw::graph::Graph;

/// Appends a directed edge unless it is already present.
///
/// The Kani body asserts the origin exists so proofs stay non-vacuous;
/// the test body downgrades that to a debug assertion.
pub(crate) fn add_edge_if_missing(graph: &mut Graph, origin: usize, target: usize, level: usize) {
    #[cfg(kani)]
    {
        let Some(node) = graph.node_mut(origin) else {
            kani::assert(false, "Kani origin node must exist");
            return;
        };
        let Some(neighbours) = node.neighbours_mut(level) else {
            kani::assert(false, "Kani origin must expose requested level");
            return;
        };
        if !neighbours.contains(&target) {
            neighbours.push(target);
        }
    }
    #[cfg(not(kani))]
    {
        let Some(node) = graph.node_mut(origin) else {
            debug_assert!(false, "missing origin node {origin}");
            return;
        };
        let Some(neighbours) = node.neighbours_mut(level) else {
            debug_assert!(false, "origin {origin} lacks requested level {level}");
            return;
        };
        if !neighbours.contains(&target) {
            neighbours.push(target);
        }
    }
}

/// Panics when the directed edge is present at the given level.
#[cfg(test)]
pub(super) fn assert_no_edge(graph: &Graph, origin: usize, target: usize, level: usize) {
    if let Some(node) = graph.node(origin)
        && level < node.level_count()
    {
        assert!(
            !node.neighbours(level).contains(&target),
            "unexpected edge {origin}->{target} at level {level}",
        );
    }
}

/// Outcome of inspecting a node pair for a mutual edge at a given level.
///
/// Modelled as a query result so callers assert on a value rather than relying
/// on a helper to panic on their behalf.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EdgeSymmetry {
    /// Both nodes list one another at the requested level.
    Symmetric,
    /// The named node is absent from the graph.
    MissingNode(usize),
    /// The named node does not expose the requested level.
    LevelAbsent { node: usize, level_count: usize },
    /// The forward edge `origin -> target` is missing.
    MissingEdge { origin: usize, target: usize },
}

/// Classifies the edge relationship between `node_a` and `node_b` at `level`.
#[cfg(test)]
pub(crate) fn edge_symmetry(
    graph: &Graph,
    node_a: usize,
    node_b: usize,
    level: usize,
) -> EdgeSymmetry {
    let Some(a) = graph.node(node_a) else {
        return EdgeSymmetry::MissingNode(node_a);
    };
    let Some(b) = graph.node(node_b) else {
        return EdgeSymmetry::MissingNode(node_b);
    };

    if level >= a.level_count() {
        return EdgeSymmetry::LevelAbsent {
            node: node_a,
            level_count: a.level_count(),
        };
    }
    if level >= b.level_count() {
        return EdgeSymmetry::LevelAbsent {
            node: node_b,
            level_count: b.level_count(),
        };
    }

    if !a.neighbours(level).contains(&node_b) {
        return EdgeSymmetry::MissingEdge {
            origin: node_a,
            target: node_b,
        };
    }
    if !b.neighbours(level).contains(&node_a) {
        return EdgeSymmetry::MissingEdge {
            origin: node_b,
            target: node_a,
        };
    }
    EdgeSymmetry::Symmetric
}

/// Asserts that two nodes reference one another at `level`.
///
/// Implemented as a macro so a failure reports the calling test's line rather
/// than a shared helper's line.
#[cfg(test)]
macro_rules! assert_bidirectional_edge {
    ($graph:expr, $node_a:expr, $node_b:expr, $level:expr $(,)?) => {{
        let symmetry =
            $crate::hnsw::insert::test_helpers::edge_symmetry($graph, $node_a, $node_b, $level);
        assert_eq!(
            symmetry,
            $crate::hnsw::insert::test_helpers::EdgeSymmetry::Symmetric,
            "expected bidirectional edge {} <-> {} at level {}",
            $node_a,
            $node_b,
            $level,
        );
    }};
}
#[cfg(test)]
pub(crate) use assert_bidirectional_edge;

#[cfg(test)]
#[derive(Debug)]
pub(super) struct TestHelpers<'graph> {
    pub(super) graph: &'graph mut Graph,
}

#[cfg(test)]
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
}

/// Describes why an edge left by [`TestHelpers::enforce_bidirectional_all`]
/// fails the reciprocity invariant.
#[cfg(test)]
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
