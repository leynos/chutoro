//! Maintains connectivity invariants when new edges are added or removed.
//!
//! Connectivity healing covers fallback linking for the newly inserted node,
//! base-layer reachability guarantees, and cleanup of evicted edges when
//! neighbour lists overflow.
//!
//! The healing process uses an iterative work queue to avoid deep recursion
//! that could cause stack overflow with pathological graph configurations.

#[cfg(not(kani))]
use std::collections::HashSet;

use super::limits::compute_connection_limit;
use super::types::{LinkContext, UpdateContext};
use crate::hnsw::graph::Graph;

/// Visited-node set for the iterative healing queues.
///
/// Production builds use a `HashSet`. Under Kani the default hasher's
/// randomized SipHash state is symbolic, which makes bounded verification of
/// any healing path intractable, so the linear-scan set below is substituted;
/// healing queues visit each node at most once, so the scan stays bounded.
#[cfg(not(kani))]
type VisitedSet = HashSet<usize>;

#[cfg(kani)]
type VisitedSet = LinearVisitedSet;

/// Linear-scan visited set standing in for `HashSet` under Kani.
///
/// Compiled under `cfg(test)` as well so its equivalence to `HashSet`
/// semantics is asserted in normal CI: the healing path itself is past the
/// tractable CBMC state space even at two nodes (a direct harness timed out
/// in symbolic execution), so this substitute is validated by test rather
/// than by proof.
#[cfg(any(kani, test))]
#[derive(Debug, Default)]
struct LinearVisitedSet {
    seen: Vec<usize>,
}

#[cfg(any(kani, test))]
impl LinearVisitedSet {
    /// Creates an empty visited set.
    #[cfg(kani)]
    fn new() -> Self {
        Self::default()
    }

    /// Inserts `id`, returning `true` when it was not already present.
    fn insert(&mut self, id: usize) -> bool {
        if self.seen.contains(&id) {
            return false;
        }
        self.seen.push(id);
        true
    }
}

#[derive(Debug)]
pub(super) struct ConnectivityHealer<'graph> {
    /// Graph whose adjacency lists are repaired during healing.
    pub(super) graph: &'graph mut Graph,
}

impl<'graph> ConnectivityHealer<'graph> {
    /// Creates a healer over the graph.
    pub(super) const fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
    }

    /// Ensures a node has base connectivity by linking it to the entry node.
    ///
    /// Uses an iterative work queue to process any nodes that become isolated
    /// due to evictions, avoiding deep recursion that could cause stack overflow.
    pub(super) fn ensure_base_connectivity(&mut self, node: usize, max_connections: usize) {
        let mut work_queue: Vec<usize> = vec![node];
        let mut visited = VisitedSet::new();

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

    /// Links the new node to the origin, dispatching on layer semantics.
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
        let mut visited = VisitedSet::new();

        while let Some(current) = work_queue.pop() {
            if let Some(evicted) = self.try_heal_node(&mut visited, current, max_connections) {
                work_queue.push(evicted);
            }
        }
    }

    /// Attempts to heal connectivity for a single node, returning any newly evicted node.
    fn try_heal_node(
        &mut self,
        visited: &mut VisitedSet,
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

    /// Inner implementation of `link_new_node` that returns the evicted node (if any)
    /// instead of recursively handling it.
    fn link_new_node_inner(&mut self, ctx: &UpdateContext, new_node: usize) -> Option<usize> {
        let limit = compute_connection_limit(ctx.level, ctx.max_connections);
        if !self.can_link_at_level(ctx.origin, ctx.level) {
            return None;
        }

        let candidate_node = self.graph.node_mut(ctx.origin)?;
        let origin_neighbours = candidate_node.neighbours_mut(ctx.level)?;
        let evicted_node = Self::add_to_neighbour_list(origin_neighbours, new_node, limit);
        if !origin_neighbours.contains(&new_node) {
            return None;
        }

        if !self.can_link_at_level(new_node, ctx.level) {
            return None;
        }

        let new_node_ref = self.graph.node_mut(new_node)?;
        let new_node_neighbours = new_node_ref.neighbours_mut(ctx.level)?;
        Self::add_to_neighbour_list(new_node_neighbours, ctx.origin, limit);
        if !new_node_neighbours.contains(&ctx.origin) {
            return None;
        }

        // Return the evicted node that needs cleanup instead of recursing
        Some(evicted_node.map_or(new_node, |node_id| {
            self.clean_up_evicted_edge_inner(node_id, ctx)
        }))
    }

    /// Cleans up a forward edge and returns the node to handle iteratively.
    fn clean_up_evicted_edge_inner(&mut self, evicted: usize, ctx: &UpdateContext) -> usize {
        let Some(evicted_node) = self.graph.node_mut(evicted) else {
            return ctx.origin; // Link succeeded to origin's perspective
        };
        if ctx.level >= evicted_node.level_count() {
            return ctx.origin;
        }

        let Some(evicted_neighbours) = evicted_node.neighbours_mut(ctx.level) else {
            return ctx.origin;
        };
        if let Some(pos) = evicted_neighbours.iter().position(|&id| id == ctx.origin) {
            evicted_neighbours.remove(pos);
        }

        if ctx.level == 0 && evicted_neighbours.is_empty() {
            evicted // Return isolated node for caller to queue
        } else {
            ctx.origin // Link succeeded
        }
    }

    /// Link a node to the entry point when other fallback candidates fail.
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

    /// Select the first fallback candidate that accepts a reciprocal link.
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

    /// Report whether a node has an initialized adjacency list for a level.
    fn can_link_at_level(&self, node_id: usize, level: usize) -> bool {
        self.graph
            .node(node_id)
            .is_some_and(|node| level < node.level_count())
    }

    /// Insert a neighbour and return the displaced tail when capacity is full.
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
#[cfg(test)]
mod tests {
    //! Equivalence coverage for the Kani visited-set substitute.

    use std::collections::HashSet;

    use rstest::rstest;

    use super::LinearVisitedSet;

    /// The linear-scan set must report insertions exactly as `HashSet` does,
    /// because Kani builds substitute it for the production `HashSet` inside
    /// the healing work queues.
    #[rstest]
    #[case::all_unique(&[1, 2, 3, 4])]
    #[case::immediate_duplicate(&[7, 7])]
    #[case::interleaved_duplicates(&[3, 1, 3, 2, 1, 3])]
    #[case::single(&[0])]
    #[case::empty(&[])]
    fn linear_set_matches_hash_set_semantics(#[case] sequence: &[usize]) {
        let mut linear = LinearVisitedSet::default();
        let mut hashed = HashSet::new();
        for &id in sequence {
            assert_eq!(
                linear.insert(id),
                hashed.insert(id),
                "insert({id}) diverged from HashSet semantics",
            );
        }
    }
}
