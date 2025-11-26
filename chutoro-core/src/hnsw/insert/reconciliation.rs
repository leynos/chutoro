//! Reconciles forward and reverse edges during insertion commit.
//!
//! Trimming can remove or reorder neighbours for existing nodes. This module
//! ensures reciprocal edges are maintained, scrubs invalid forward edges, and
//! restores base connectivity when nodes become isolated at the base layer.

use crate::hnsw::graph::Graph;

use super::{
    connectivity::ConnectivityHealer, limits::compute_connection_limit, types::UpdateContext,
};

#[derive(Debug)]
pub(super) struct EdgeReconciler<'graph> {
    pub(super) graph: &'graph mut Graph,
}

impl<'graph> EdgeReconciler<'graph> {
    pub(super) fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
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
            evicted = neighbours.pop();
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
            self.scrub_forward_edge(ctx, target, evicted);
        }

        true
    }

    pub(super) fn scrub_forward_edge(
        &mut self,
        ctx: &UpdateContext,
        target: usize,
        evicted: usize,
    ) {
        let evicted_ctx = UpdateContext {
            origin: evicted,
            level: ctx.level,
            max_connections: ctx.max_connections,
        };
        self.remove_forward_edge_from(&evicted_ctx, target);
    }

    pub(super) fn remove_forward_edge_from(&mut self, ctx: &UpdateContext, target: usize) {
        let Some(origin_node) = self.graph.node_mut(ctx.origin) else {
            return;
        };
        if ctx.level >= origin_node.level_count() {
            return;
        }

        let neighbours = origin_node.neighbours_mut(ctx.level);
        if let Some(pos) = neighbours.iter().position(|&id| id == target) {
            neighbours.remove(pos);
            if ctx.level == 0 && neighbours.is_empty() {
                let mut healer = ConnectivityHealer::new(self.graph);
                healer.ensure_base_connectivity(ctx.origin, ctx.max_connections);
            }
        }
    }
}
