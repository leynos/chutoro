//! Applies staged neighbour updates to the graph during insertion commit.
//!
//! This module is responsible for writing trimmed neighbour lists back to the
//! graph, both for the newly inserted node and for existing nodes touched by
//! the insertion. Reconciliation of forward and reverse edges is delegated to
//! [`EdgeReconciler`] to keep responsibilities focused.

use crate::hnsw::{error::HnswError, graph::Graph};

use super::{
    reconciliation::EdgeReconciler,
    types::{FinalisedUpdate, NewNodeContext, UpdateContext},
};

pub(super) type ApplyUpdatesOutcome = (Vec<Vec<usize>>, Vec<(usize, usize)>);

#[derive(Debug)]
pub(super) struct CommitApplicator<'graph> {
    pub(super) graph: &'graph mut Graph,
}

impl<'graph> CommitApplicator<'graph> {
    pub(super) fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
    }

    /// Writes the filtered neighbour lists back to the newly attached node.
    pub(super) fn apply_new_node_neighbours(
        &mut self,
        node_id: usize,
        node_level: usize,
        filtered_neighbours: Vec<Vec<usize>>,
    ) -> Result<(), HnswError> {
        let Some(node_ref) = self.graph.node_mut(node_id) else {
            return Err(HnswError::GraphInvariantViolation {
                message: format!("node {node_id} missing after attach during commit"),
            });
        };
        for (level, neighbours) in filtered_neighbours
            .into_iter()
            .enumerate()
            .take(node_level + 1)
        {
            let list = node_ref.neighbours_mut(level);
            list.clear();
            list.extend(neighbours);
        }
        Ok(())
    }

    /// Applies the neighbour updates gathered during staging to the existing
    /// nodes now that their adjacency lists have been trimmed.
    pub(super) fn apply_neighbour_updates(
        &mut self,
        final_updates: Vec<FinalisedUpdate>,
        max_connections: usize,
        new_node: NewNodeContext,
    ) -> Result<ApplyUpdatesOutcome, HnswError> {
        let mut reciprocated: Vec<Vec<usize>> = vec![Vec::new(); new_node.level + 1];
        let mut touched: Vec<(usize, usize)> = Vec::with_capacity(final_updates.len());
        let mut reconciler = EdgeReconciler::new(self.graph);

        for (update, neighbours) in final_updates {
            let level = update.ctx.level;
            let previous = reconciler
                .graph()
                .node(update.node)
                .map(|node| node.neighbours(level).to_vec())
                .ok_or_else(|| HnswError::GraphInvariantViolation {
                    message: format!("node {} missing during insertion commit", update.node),
                })?;

            let mut next = neighbours;
            let ctx = UpdateContext {
                origin: update.node,
                level,
                max_connections,
            };

            reconciler.reconcile_removed_edges(&ctx, &previous, &next);
            reconciler.reconcile_added_edges(&ctx, &mut next);

            if level <= new_node.level && next.contains(&new_node.id) {
                reciprocated[level].push(update.node);
            }

            let node_ref = reconciler
                .graph_mut()
                .node_mut(update.node)
                .ok_or_else(|| HnswError::GraphInvariantViolation {
                    message: format!("node {} missing during insertion commit", update.node),
                })?;
            let list = node_ref.neighbours_mut(level);
            list.clear();
            list.extend(next);

            touched.push((update.node, level));
        }

        Ok((reciprocated, touched))
    }
}
