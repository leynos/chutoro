//! Maintains connectivity invariants when new edges are added or removed.
//!
//! Connectivity healing covers fallback linking for the newly inserted node,
//! base-layer reachability guarantees, and cleanup of evicted edges when
//! neighbour lists overflow.
//!
//! The healing process uses an iterative work queue to avoid deep recursion
//! that could cause stack overflow with pathological graph configurations.

use std::collections::HashSet;

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

    /// Ensures a node has base connectivity by linking it to the entry node.
    ///
    /// Uses an iterative work queue to process any nodes that become isolated
    /// due to evictions, avoiding deep recursion that could cause stack overflow.
    pub(super) fn ensure_base_connectivity(&mut self, node: usize, max_connections: usize) {
        let mut work_queue: Vec<usize> = vec![node];
        let mut visited: HashSet<usize> = HashSet::new();

        while let Some(current) = work_queue.pop() {
            if !visited.insert(current) {
                continue;
            }

            let Some(entry) = self.graph.entry() else {
                continue;
            };

            if entry.node == current {
                continue;
            }

            let ctx = UpdateContext {
                origin: entry.node,
                level: 0,
                max_connections,
            };

            if let Some(evicted) = self.link_new_node_inner(&ctx, current) {
                work_queue.push(evicted);
            }
        }
    }

    pub(super) fn link_new_node(&mut self, ctx: &UpdateContext, new_node: usize) -> bool {
        if ctx.level == 0 {
            self.link_new_node_base_layer(ctx, new_node)
        } else {
            self.link_new_node_upper_layer(ctx, new_node)
        }
    }

    /// Handles base layer (level 0) linking with iterative eviction processing.
    fn link_new_node_base_layer(&mut self, ctx: &UpdateContext, new_node: usize) -> bool {
        let result = self.link_new_node_inner(ctx, new_node);
        if let Some(evicted) = result {
            self.process_eviction_queue(evicted, ctx.max_connections);
        }

        result.is_some() || self.node_has_link(new_node, ctx.origin, 0)
    }

    /// Handles upper layer linking.
    fn link_new_node_upper_layer(&mut self, ctx: &UpdateContext, new_node: usize) -> bool {
        self.link_new_node_inner(ctx, new_node).is_some()
            || self.node_has_link(new_node, ctx.origin, ctx.level)
    }

    /// Processes evicted nodes iteratively to restore their connectivity.
    fn process_eviction_queue(&mut self, initial: usize, max_connections: usize) {
        let mut work_queue: Vec<usize> = vec![initial];
        let mut visited: HashSet<usize> = HashSet::new();

        while let Some(current) = work_queue.pop() {
            if let Some(evicted) = self.try_heal_node(&mut visited, current, max_connections) {
                work_queue.push(evicted);
            }
        }
    }

    /// Attempts to heal connectivity for a single node, returning any newly evicted node.
    fn try_heal_node(
        &mut self,
        visited: &mut HashSet<usize>,
        current: usize,
        max_connections: usize,
    ) -> Option<usize> {
        if !visited.insert(current) {
            return None;
        }

        let entry = self.graph.entry()?;
        if entry.node == current {
            return None;
        }

        let heal_ctx = UpdateContext {
            origin: entry.node,
            level: 0,
            max_connections,
        };

        self.link_new_node_inner(&heal_ctx, current)
    }

    /// Checks if a node has a link to a target at a given level.
    fn node_has_link(&self, node: usize, target: usize, level: usize) -> bool {
        self.graph
            .node(node)
            .is_some_and(|n| level < n.level_count() && n.neighbours(level).contains(&target))
    }

    /// Inner implementation of link_new_node that returns the evicted node (if any)
    /// instead of recursively handling it.
    fn link_new_node_inner(&mut self, ctx: &UpdateContext, new_node: usize) -> Option<usize> {
        let limit = compute_connection_limit(ctx.level, ctx.max_connections);
        if !self.can_link_at_level(ctx.origin, ctx.level) {
            return None;
        }

        let candidate_node = self.graph.node_mut(ctx.origin)?;
        let neighbours = candidate_node.neighbours_mut(ctx.level);
        let evicted = Self::add_to_neighbour_list(neighbours, new_node, limit);
        if !neighbours.contains(&new_node) {
            return None;
        }

        if !self.can_link_at_level(new_node, ctx.level) {
            return None;
        }

        let new_node_ref = self.graph.node_mut(new_node)?;
        let neighbours = new_node_ref.neighbours_mut(ctx.level);
        Self::add_to_neighbour_list(neighbours, ctx.origin, limit);
        if !neighbours.contains(&ctx.origin) {
            return None;
        }

        // Return the evicted node that needs cleanup instead of recursing
        if let Some(evicted) = evicted {
            self.clean_up_evicted_edge_inner(evicted, ctx)
        } else {
            // Link succeeded, no eviction
            Some(new_node)
        }
    }

    /// Cleans up a forward edge from an evicted node and returns the evicted node
    /// if it became isolated (for caller to handle iteratively).
    fn clean_up_evicted_edge_inner(
        &mut self,
        evicted: usize,
        ctx: &UpdateContext,
    ) -> Option<usize> {
        let Some(evicted_node) = self.graph.node_mut(evicted) else {
            return Some(ctx.origin); // Link succeeded to origin's perspective
        };
        if ctx.level >= evicted_node.level_count() {
            return Some(ctx.origin);
        }

        let evicted_neighbours = evicted_node.neighbours_mut(ctx.level);
        if let Some(pos) = evicted_neighbours.iter().position(|&id| id == ctx.origin) {
            evicted_neighbours.remove(pos);
        }

        if ctx.level == 0 && evicted_neighbours.is_empty() {
            Some(evicted) // Return isolated node for caller to queue
        } else {
            Some(ctx.origin) // Link succeeded
        }
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
        fallback: Option<&[usize]>,
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
            .is_some_and(|node| level < node.level_count())
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
}
