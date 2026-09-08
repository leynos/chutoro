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

/// Reconciles forward and reverse edges while committing one insertion.
#[derive(Debug)]
pub(super) struct EdgeReconciler<'graph> {
    /// Graph mutated during insertion reconciliation.
    pub(super) graph: &'graph mut Graph,
    /// Edge removals deferred until all staged updates complete.
    deferred_scrubs: Vec<DeferredScrub>,
}

impl<'graph> EdgeReconciler<'graph> {
    /// Creates a reconciler over the graph with no scrubs pending.
    pub(super) const fn new(graph: &'graph mut Graph) -> Self {
        Self {
            graph,
            deferred_scrubs: Vec::new(),
        }
    }

    /// Returns mutable access to the underlying graph.
    pub(super) const fn graph_mut(&mut self) -> &mut Graph {
        self.graph
    }

    /// Returns shared access to the underlying graph.
    pub(super) const fn graph(&self) -> &Graph {
        self.graph
    }

    /// Removes reciprocal edges for neighbours dropped from the origin.
    ///
    /// Runs after the origin's list write-back so base-layer healing of any
    /// newly isolated node observes final state rather than the stale list.
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
            if self.remove_reverse_edge(ctx, target) {
                isolated.push(target);
            }
        }

        if isolated.is_empty() {
            return;
        }

        tracing::debug!(
            operation = "reconcile_removed_edges",
            level = ctx.level,
            isolated_count = isolated.len(),
            "healing base connectivity for nodes isolated by removed edges"
        );
        #[cfg(feature = "metrics")]
        metrics::counter!("chutoro.hnsw.reconciliation.healed_nodes_total")
            .increment(isolated.len() as u64);
        let mut healer = ConnectivityHealer::new(self.graph);
        for node in isolated {
            healer.ensure_base_connectivity(node, ctx.max_connections);
        }
    }

    /// Removes the target's reciprocal edge to the origin.
    ///
    /// Returns whether the removal leaves a base-layer target isolated, so the
    /// caller can defer healing until all removals are complete.
    fn remove_reverse_edge(&mut self, ctx: &UpdateContext, target: usize) -> bool {
        let Some(target_node) = self.graph.node_mut(target) else {
            return false;
        };
        if ctx.level >= target_node.level_count() {
            return false;
        }

        let Some(neighbours) = target_node.neighbours_mut(ctx.level) else {
            return false;
        };
        let Some(pos) = neighbours.iter().position(|&id| id == ctx.origin) else {
            return false;
        };

        neighbours.remove(pos);
        ctx.level == 0 && neighbours.is_empty()
    }

    /// Retains only targets whose reverse edge could be ensured.
    pub(super) fn reconcile_added_edges(&mut self, ctx: &UpdateContext, next: &mut Vec<usize>) {
        next.retain(|&target| self.ensure_reverse_edge(ctx, target));
    }

    /// Ensures `target` links back to the origin, evicting if at capacity.
    ///
    /// Returns `false` when the target is missing or lacks the level. An
    /// eviction defers a scrub for the orphaned forward edge.
    pub(super) fn ensure_reverse_edge(&mut self, ctx: &UpdateContext, target: usize) -> bool {
        let Some(target_node) = self.graph.node_mut(target) else {
            return false;
        };
        if ctx.level >= target_node.level_count() {
            return false;
        }

        let limit = compute_connection_limit(ctx.level, ctx.max_connections);
        let Some(neighbours) = target_node.neighbours_mut(ctx.level) else {
            return false;
        };
        if neighbours.contains(&ctx.origin) {
            return true;
        }

        let mut evicted_origin: Option<usize> = None;
        if neighbours.len() < limit {
            neighbours.push(ctx.origin);
        } else if !neighbours.is_empty() {
            // Neighbour lists produced by trimming are ordered furthest-first; evict
            // the furthest (front) to preserve closer entries when capacity is full.
            evicted_origin = Some(neighbours.remove(0));
            neighbours.push(ctx.origin);
        }

        #[cfg(test)]
        {
            assert!(
                neighbours.contains(&ctx.origin),
                "ensure_reverse_edge failed to insert {origin}->{target} at level {level}; degree {} (limit {limit})",
                neighbours.len(),
                origin = ctx.origin,
                target = target,
                level = ctx.level,
            );
        }

        if let Some(evicted) = evicted_origin {
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

            tracing::debug!(
                operation = "apply_deferred_scrubs",
                level = scrub.level,
                "scrubbing orphaned forward edge left by an eviction"
            );
            #[cfg(feature = "metrics")]
            metrics::counter!(
                "chutoro.hnsw.reconciliation.orphan_scrubs_total",
                "layer" => if scrub.level == 0 { "base" } else { "upper" }
            )
            .increment(1);
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
    const fn should_heal_connectivity(
        initial_len: usize,
        neighbours: &[usize],
        level: usize,
    ) -> bool {
        let neighbour_was_removed = initial_len != neighbours.len();
        let is_base_layer = level == 0;
        let is_now_isolated = neighbours.is_empty();

        neighbour_was_removed && is_base_layer && is_now_isolated
    }

    /// Removes the origin's forward edge to `target`, healing base-layer
    /// isolation the removal causes.
    pub(super) fn remove_forward_edge_from(&mut self, ctx: &UpdateContext, target: usize) {
        let Some(origin_node) = self.graph.node_mut(ctx.origin) else {
            return;
        };
        if ctx.level >= origin_node.level_count() {
            return;
        }

        let Some(neighbours) = origin_node.neighbours_mut(ctx.level) else {
            return;
        };
        let initial_len = neighbours.len();
        if let Some(pos) = neighbours.iter().position(|&id| id == target) {
            neighbours.remove(pos);
            if Self::should_heal_connectivity(initial_len, neighbours, ctx.level) {
                let mut healer = ConnectivityHealer::new(self.graph);
                healer.ensure_base_connectivity(ctx.origin, ctx.max_connections);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests reverse-edge removal during insertion reconciliation.

    use super::{EdgeReconciler, UpdateContext};
    use crate::hnsw::{
        error::HnswError,
        graph::{Graph, NodeContext},
        params::HnswParams,
    };

    fn graph_with_two_nodes(level: usize) -> Result<Graph, HnswError> {
        let params = HnswParams::new(2, 4)?;
        let mut graph = Graph::with_capacity(params, 2);
        graph.insert_first(NodeContext {
            node: 0,
            level,
            sequence: 0,
        })?;
        graph.attach_node(NodeContext {
            node: 1,
            level,
            sequence: 1,
        })?;
        Ok(graph)
    }

    fn update_context(level: usize) -> UpdateContext {
        UpdateContext {
            origin: 0,
            level,
            max_connections: 2,
        }
    }

    #[test]
    fn remove_reverse_edge_reports_a_base_layer_target_it_isolates() {
        let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
        graph
            .node_mut(1)
            .expect("target node must be present")
            .neighbours_mut(0)
            .expect("target must have a base layer")
            .push(0);

        let mut reconciler = EdgeReconciler::new(&mut graph);

        assert!(reconciler.remove_reverse_edge(&update_context(0), 1));
        assert!(
            reconciler
                .graph
                .node(1)
                .expect("target node must be present")
                .neighbours(0)
                .is_empty()
        );
    }

    #[test]
    fn remove_reverse_edge_does_not_report_an_upper_layer_target_as_isolated() {
        let mut graph = graph_with_two_nodes(1).expect("test graph must be valid");
        graph
            .node_mut(1)
            .expect("target node must be present")
            .neighbours_mut(1)
            .expect("target must have an upper layer")
            .push(0);

        let mut reconciler = EdgeReconciler::new(&mut graph);

        assert!(!reconciler.remove_reverse_edge(&update_context(1), 1));
        assert!(
            reconciler
                .graph
                .node(1)
                .expect("target node must be present")
                .neighbours(1)
                .is_empty()
        );
    }

    #[test]
    fn remove_reverse_edge_returns_false_when_the_reverse_edge_is_missing() {
        let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
        let mut reconciler = EdgeReconciler::new(&mut graph);

        assert!(!reconciler.remove_reverse_edge(&update_context(0), 1));
    }

    #[test]
    fn remove_reverse_edge_returns_false_when_target_or_level_is_missing() {
        let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
        let mut reconciler = EdgeReconciler::new(&mut graph);

        assert!(!reconciler.remove_reverse_edge(&update_context(0), 2));
        assert!(!reconciler.remove_reverse_edge(&update_context(1), 1));
    }
}
