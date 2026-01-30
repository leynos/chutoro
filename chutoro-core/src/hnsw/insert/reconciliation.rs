//! Reconciles forward and reverse edges during insertion commit.
//!
//! Trimming can remove or reorder neighbours for existing nodes. This module
//! ensures reciprocal edges are maintained, scrubs invalid forward edges, and
//! restores base connectivity when nodes become isolated at the base layer.
//!
//! Scrub requests are deferred to avoid conflicts when multiple updates in the
//! same batch touch overlapping edges. After all updates are processed, the
//! deferred scrubs are filtered against the final edge set to ensure we don't
//! remove edges that were added by later updates.

use crate::hnsw::graph::Graph;

use super::{
    connectivity::ConnectivityHealer,
    limits::compute_connection_limit,
    types::{DeferredScrub, UpdateContext},
};

#[derive(Debug)]
pub(super) struct EdgeReconciler<'graph> {
    pub(super) graph: &'graph mut Graph,
    deferred_scrubs: Vec<DeferredScrub>,
}

impl<'graph> EdgeReconciler<'graph> {
    pub(super) fn new(graph: &'graph mut Graph) -> Self {
        Self {
            graph,
            deferred_scrubs: Vec::new(),
        }
    }

    pub(super) fn graph_mut(&mut self) -> &mut Graph {
        self.graph
    }

    pub(super) fn graph(&self) -> &Graph {
        self.graph
    }

    pub(super) fn reconcile_removed_edges(
        &mut self,
        ctx: &UpdateContext,
        previous: &[usize],
        next: &[usize],
    ) {
        let mut isolated: Vec<usize> = Vec::new();
        for &target in previous {
            if next.contains(&target) {
                continue;
            }
            let Some(target_node) = self.graph.node_mut(target) else {
                continue;
            };
            if ctx.level >= target_node.level_count() {
                continue;
            }

            let neighbours = target_node.neighbours_mut(ctx.level);
            let Some(pos) = neighbours.iter().position(|&id| id == ctx.origin) else {
                continue;
            };

            neighbours.remove(pos);
            if ctx.level == 0 && neighbours.is_empty() {
                isolated.push(target);
            }
        }

        if isolated.is_empty() {
            return;
        }

        let mut healer = ConnectivityHealer::new(self.graph);
        for node in isolated {
            healer.ensure_base_connectivity(node, ctx.max_connections);
        }
    }

    pub(super) fn reconcile_added_edges(&mut self, ctx: &UpdateContext, next: &mut Vec<usize>) {
        next.retain(|&target| self.ensure_reverse_edge(ctx, target));
    }

    pub(super) fn ensure_reverse_edge(&mut self, ctx: &UpdateContext, target: usize) -> bool {
        let Some(target_node) = self.graph.node_mut(target) else {
            return false;
        };
        if ctx.level >= target_node.level_count() {
            return false;
        }

        let limit = compute_connection_limit(ctx.level, ctx.max_connections);
        let neighbours = target_node.neighbours_mut(ctx.level);
        if neighbours.contains(&ctx.origin) {
            return true;
        }

        let mut evicted: Option<usize> = None;
        if neighbours.len() < limit {
            neighbours.push(ctx.origin);
        } else if !neighbours.is_empty() {
            // Neighbour lists produced by trimming are ordered furthest-first; evict
            // the furthest (front) to preserve closer entries when capacity is full.
            evicted = Some(neighbours.remove(0));
            neighbours.push(ctx.origin);
        }

        #[cfg(test)]
        {
            if !neighbours.contains(&ctx.origin) {
                panic!(
                    "ensure_reverse_edge failed to insert {origin}->{target} at level {level}; degree {} (limit {limit})",
                    neighbours.len(),
                    origin = ctx.origin,
                    target = target,
                    level = ctx.level,
                );
            }
        }

        if let Some(evicted) = evicted {
            self.deferred_scrubs.push(DeferredScrub {
                origin: evicted,
                target,
                level: ctx.level,
            });
        }

        true
    }

    /// Applies all deferred scrubs, removing one-way edges where the reverse
    /// edge was not restored by a later update.
    ///
    /// For each scrub (origin evicted from target), we check:
    /// 1. If target now links back to origin, reciprocity is intact - skip
    /// 2. If origin no longer links to target, the edge is already gone - skip
    /// 3. Otherwise, the forward edge is orphaned - remove it
    pub(super) fn apply_deferred_scrubs(&mut self, max_connections: usize) {
        let scrubs = std::mem::take(&mut self.deferred_scrubs);
        for scrub in scrubs {
            // Check if target now has a forward link back to origin (i.e.,
            // a later update re-added the reciprocal edge).
            let target_links_origin = self
                .graph
                .node(scrub.target)
                .and_then(|node| {
                    (scrub.level < node.level_count())
                        .then(|| node.neighbours(scrub.level).contains(&scrub.origin))
                })
                .unwrap_or(false);

            if target_links_origin {
                continue;
            }

            let ctx = UpdateContext {
                origin: scrub.origin,
                level: scrub.level,
                max_connections,
            };
            self.remove_forward_edge_from(&ctx, scrub.target);
        }
    }

    /// Returns true if connectivity healing should be triggered after removing
    /// a neighbour. Healing is needed when a node becomes isolated at the base
    /// layer after a successful removal.
    fn should_heal_connectivity(initial_len: usize, neighbours: &[usize], level: usize) -> bool {
        let neighbour_was_removed = initial_len != neighbours.len();
        let is_base_layer = level == 0;
        let is_now_isolated = neighbours.is_empty();

        neighbour_was_removed && is_base_layer && is_now_isolated
    }

    pub(super) fn remove_forward_edge_from(&mut self, ctx: &UpdateContext, target: usize) {
        let Some(origin_node) = self.graph.node_mut(ctx.origin) else {
            return;
        };
        if ctx.level >= origin_node.level_count() {
            return;
        }

        let neighbours = origin_node.neighbours_mut(ctx.level);
        let initial_len = neighbours.len();
        neighbours.retain(|&id| id != target);
        if Self::should_heal_connectivity(initial_len, neighbours, ctx.level) {
            let mut healer = ConnectivityHealer::new(self.graph);
            healer.ensure_base_connectivity(ctx.origin, ctx.max_connections);
        }
    }
}
