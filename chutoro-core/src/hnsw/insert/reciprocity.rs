//! Ensures reciprocal edges for nodes touched during insertion.
//!
//! Reciprocity enforcement cleans up one-way edges that can emerge after
//! trimming and fallback linking. It also provides a workspace to adjust the
//! new node's neighbour lists based on which connections were successfully
//! reciprocated.

use std::collections::HashSet;

use crate::hnsw::graph::Graph;

use super::{limits::compute_connection_limit, types::FinalisedUpdate};

#[cfg(any(test, debug_assertions))]
#[derive(Debug)]
pub(super) struct ReciprocityAuditor<'graph> {
    graph: &'graph Graph,
}

#[cfg(any(test, debug_assertions))]
#[derive(Clone, Copy)]
struct AuditContext {
    level: usize,
    max_connections: usize,
}

#[cfg(any(test, debug_assertions))]
impl<'graph> ReciprocityAuditor<'graph> {
    pub(super) fn new(graph: &'graph Graph) -> Self {
        Self { graph }
    }

    pub(super) fn assert_reciprocity_for_touched(
        &self,
        touched: &[(usize, usize)],
        max_connections: usize,
    ) {
        let mut seen = HashSet::new();
        for &(origin, level) in touched {
            if !seen.insert((origin, level)) {
                continue;
            }

            let ctx = AuditContext {
                level,
                max_connections,
            };
            self.assert_origin_state(origin, ctx);
        }
    }

    fn assert_origin_state(&self, origin: usize, ctx: AuditContext) {
        let level = ctx.level;
        let Some(origin_node) = self.graph.node(origin) else {
            panic!("reciprocity audit: node {origin} missing from graph");
        };
        if level >= origin_node.level_count() {
            return;
        }

        let origin_neighbours = origin_node.neighbours(level);
        let origin_limit = compute_connection_limit(level, ctx.max_connections);
        assert!(
            origin_neighbours.len() <= origin_limit,
            "reciprocity audit: node {origin} exceeds degree limit {origin_limit} at level \
             {level}; neighbours {:?}",
            origin_neighbours,
        );

        for &target in origin_neighbours {
            self.assert_target_state(origin, target, ctx);
        }
    }

    fn assert_target_state(&self, origin: usize, target: usize, ctx: AuditContext) {
        let level = ctx.level;
        let Some(target_node) = self.graph.node(target) else {
            panic!(
                "reciprocity audit: edge {origin}->{target} at level {level} targets missing node",
            );
        };

        let target_levels = target_node.level_count();
        assert!(
            ctx.level < target_levels,
            "reciprocity audit: edge {origin}->{target} points to absent level {level} \
             (target exposes {target_levels} levels)",
        );

        let neighbours = target_node.neighbours(ctx.level);
        let limit = compute_connection_limit(ctx.level, ctx.max_connections);
        assert!(
            neighbours.contains(&origin),
            "reciprocity audit: missing reverse edge {target}->{origin} at level {level}; \
             target degree {} (limit {limit})",
            neighbours.len(),
        );
        assert!(
            neighbours.len() <= limit,
            "reciprocity audit: node {target} exceeds degree limit {limit} at level {level}; \
             neighbours {:?}",
            neighbours,
        );
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
                // Fallback prioritises connectivity over preserving any prior
                // ordering when no capacity remains; evict the tail entry to
                // guarantee a reciprocal link for the new node.
                neighbour_list.pop();
                neighbour_list.push(self.new_node);
                return Some(candidate);
            }
        }

        None
    }
}
