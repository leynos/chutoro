//! Maintains connectivity invariants when new edges are added or removed.
//!
//! Connectivity healing covers fallback linking for the newly inserted node,
//! base-layer reachability guarantees, and cleanup of evicted edges when
//! neighbour lists overflow.

use super::limits::compute_connection_limit;
use super::types::{LinkContext, UpdateContext};
use crate::hnsw::graph::Graph;

#[derive(Debug)]
pub(super) struct ConnectivityHealer<'graph> {
    pub(super) graph: &'graph mut Graph,
}

impl<'graph> ConnectivityHealer<'graph> {
    pub(super) fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
    }

    pub(super) fn ensure_base_connectivity(&mut self, node: usize, max_connections: usize) {
        if let Some(entry) = self.graph.entry() {
            if entry.node == node {
                return;
            }

            let ctx = UpdateContext {
                origin: entry.node,
                level: 0,
                max_connections,
            };

            let _ = self.link_new_node(&ctx, node);
        }
    }

    pub(super) fn link_new_node(&mut self, ctx: &UpdateContext, new_node: usize) -> bool {
        let limit = compute_connection_limit(ctx.level, ctx.max_connections);
        if !self.can_link_at_level(ctx.origin, ctx.level) {
            return false;
        }

        let Some(candidate_node) = self.graph.node_mut(ctx.origin) else {
            return false;
        };

        let neighbours = candidate_node.neighbours_mut(ctx.level);
        let evicted = Self::add_to_neighbour_list(neighbours, new_node, limit);
        if !neighbours.contains(&new_node) {
            return false;
        }

        if !self.can_link_at_level(new_node, ctx.level) {
            return false;
        }

        let Some(new_node_ref) = self.graph.node_mut(new_node) else {
            return false;
        };

        let neighbours = new_node_ref.neighbours_mut(ctx.level);
        let limit_new = compute_connection_limit(ctx.level, ctx.max_connections);
        Self::add_to_neighbour_list(neighbours, ctx.origin, limit_new);
        if !neighbours.contains(&ctx.origin) {
            return false;
        }

        if let Some(evicted) = evicted {
            self.clean_up_evicted_edge(evicted, ctx);
        }
        true
    }

    pub(super) fn attach_entry_fallback(
        &mut self,
        level: usize,
        max_connections: usize,
        new_node: usize,
    ) -> Option<usize> {
        self.graph.entry().and_then(|entry| {
            let ctx = UpdateContext {
                origin: entry.node,
                level,
                max_connections,
            };
            self.link_new_node(&ctx, new_node).then_some(entry.node)
        })
    }

    pub(super) fn select_new_node_fallback(
        &mut self,
        ctx: LinkContext,
        fallback: Option<&Vec<usize>>,
    ) -> Option<usize> {
        let linked = fallback
            .into_iter()
            .flat_map(|candidates| candidates.iter().copied())
            .find(|&candidate| {
                let link = UpdateContext {
                    origin: candidate,
                    level: ctx.level,
                    max_connections: ctx.max_connections,
                };
                self.link_new_node(&link, ctx.new_node)
            });

        linked.or_else(|| self.attach_entry_fallback(ctx.level, ctx.max_connections, ctx.new_node))
    }

    fn can_link_at_level(&self, node_id: usize, level: usize) -> bool {
        self.graph
            .node(node_id)
            .map(|node| level < node.level_count())
            .unwrap_or(false)
    }

    fn add_to_neighbour_list(
        neighbours: &mut Vec<usize>,
        new_id: usize,
        limit: usize,
    ) -> Option<usize> {
        if neighbours.contains(&new_id) {
            return None;
        }
        if neighbours.len() < limit {
            neighbours.push(new_id);
            return None;
        }
        if let Some(evicted) = neighbours.pop() {
            neighbours.push(new_id);
            return Some(evicted);
        }
        None
    }

    fn clean_up_evicted_edge(&mut self, evicted: usize, ctx: &UpdateContext) {
        let Some(evicted_node) = self.graph.node_mut(evicted) else {
            return;
        };
        if ctx.level >= evicted_node.level_count() {
            return;
        }

        let evicted_neighbours = evicted_node.neighbours_mut(ctx.level);
        if let Some(pos) = evicted_neighbours.iter().position(|&id| id == ctx.origin) {
            evicted_neighbours.remove(pos);
        }
        if ctx.level == 0 && evicted_neighbours.is_empty() {
            self.ensure_base_connectivity(evicted, ctx.max_connections);
        }
    }
}
