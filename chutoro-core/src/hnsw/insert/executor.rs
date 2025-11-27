//! Applies staged HNSW insertions by mutating the graph and scheduling trim
//! jobs.

use std::collections::HashMap;

use crate::hnsw::{
    error::HnswError,
    graph::{ApplyContext, Graph, NodeContext},
};

use super::commit::CommitApplicator;
use super::connectivity::ConnectivityHealer;
use super::limits::compute_connection_limit;
use super::reciprocity::{ReciprocityEnforcer, ReciprocityWorkspace};
use super::staging::InsertionStager;
use super::types::{
    FinalisedUpdate, HealingContext, LinkContext, NewNodeContext, PreparedInsertion, TrimWork,
};

pub(crate) use super::types::{TrimJob, TrimResult};

#[derive(Debug)]
pub(crate) struct InsertionExecutor<'graph> {
    graph: &'graph mut Graph,
}

impl<'graph> InsertionExecutor<'graph> {
    pub(crate) fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
    }

    /// Prepares an insertion commit by staging all neighbour list updates.
    ///
    /// The returned [`PreparedInsertion`] captures the new node metadata and
    /// the neighbour lists for all affected nodes as they would appear after
    /// linking. Trimming is deferred to the caller, which should compute
    /// distances without holding the graph lock and then call
    /// [`InsertionExecutor::commit`] with the resulting [`TrimResult`]s.
    pub(crate) fn apply(
        &mut self,
        node: NodeContext,
        apply_ctx: ApplyContext<'_>,
    ) -> Result<(PreparedInsertion, Vec<TrimJob>), HnswError> {
        let ApplyContext { params, plan } = apply_ctx;
        let NodeContext {
            node,
            level,
            sequence,
        } = node;

        let stager = InsertionStager::new(&*self.graph);
        stager.ensure_slot_available(node)?;

        let promote_entry = level > self.graph.entry().map(|entry| entry.level).unwrap_or(0);
        let max_connections = params.max_connections();
        let (mut new_node_neighbours, staged, _initialised, needs_trim) = stager
            .process_insertion_layers(
                NodeContext {
                    node,
                    level,
                    sequence,
                },
                plan,
                max_connections,
            )?;
        InsertionStager::dedupe_new_node_lists(&mut new_node_neighbours);
        let (updates, trim_jobs) = stager.generate_updates_and_trim_jobs(
            NodeContext {
                node,
                level,
                sequence,
            },
            TrimWork {
                staged,
                needs_trim,
                max_connections,
            },
        )?;

        Ok((
            PreparedInsertion {
                node: NodeContext {
                    node,
                    level,
                    sequence,
                },
                promote_entry,
                new_node_neighbours,
                updates,
                max_connections,
            },
            trim_jobs,
        ))
    }

    /// Applies a prepared insertion after trim distances have been evaluated.
    pub(crate) fn commit(
        &mut self,
        prepared: PreparedInsertion,
        trims: Vec<TrimResult>,
    ) -> Result<(), HnswError> {
        let PreparedInsertion {
            node,
            promote_entry,
            new_node_neighbours,
            updates,
            max_connections,
        } = prepared;

        let new_node = NewNodeContext {
            id: node.node,
            level: node.level,
        };

        let mut final_updates = Self::prepare_final_updates(updates, trims);

        let mut filtered_new_node_neighbours = new_node_neighbours.clone();
        ReciprocityWorkspace {
            filtered: &mut filtered_new_node_neighbours,
            original: &new_node_neighbours,
            final_updates: &mut final_updates,
            new_node: new_node.id,
            max_connections,
        }
        .apply();

        self.graph.attach_node(node)?;

        let (mut reciprocated, mut touched) = {
            let mut applicator = CommitApplicator::new(self.graph);
            applicator.apply_neighbour_updates(final_updates, max_connections, new_node)?
        };

        self.heal_connectivity_gaps(
            &mut reciprocated,
            HealingContext {
                filtered_new_node_neighbours: &filtered_new_node_neighbours,
                new_node_id: new_node.id,
                max_connections,
            },
        );

        {
            let mut applicator = CommitApplicator::new(self.graph);
            applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;
        }

        touched.extend((0..=new_node.level).map(|level| (new_node.id, level)));

        {
            let mut enforcer = ReciprocityEnforcer::new(self.graph);
            enforcer.ensure_reciprocity_for_touched(&touched, max_connections);
        }

        if promote_entry {
            self.graph.promote_entry(new_node.id, new_node.level);
        }

        Ok(())
    }

    fn prepare_final_updates(
        updates: Vec<super::types::StagedUpdate>,
        trims: Vec<TrimResult>,
    ) -> Vec<FinalisedUpdate> {
        let mut trim_lookup: HashMap<(usize, usize), Vec<usize>> = trims
            .into_iter()
            .map(|result| ((result.node, result.ctx.level), result.neighbours))
            .collect();

        let mut final_updates: Vec<FinalisedUpdate> = Vec::with_capacity(updates.len());
        for update in updates {
            let neighbours = trim_lookup
                .remove(&(update.node, update.ctx.level))
                .unwrap_or_else(|| update.candidates.clone());
            final_updates.push((update, neighbours));
        }

        final_updates
    }

    fn heal_connectivity_gaps(
        &mut self,
        reciprocated: &mut [Vec<usize>],
        healing_ctx: HealingContext<'_>,
    ) {
        let mut healer = ConnectivityHealer::new(self.graph);
        for (level, neighbours) in reciprocated.iter_mut().enumerate() {
            neighbours.sort_unstable();
            neighbours.dedup();
            let limit = compute_connection_limit(level, healing_ctx.max_connections);
            if neighbours.len() > limit {
                neighbours.truncate(limit);
            }
            if !neighbours.is_empty() {
                continue;
            }

            let link_ctx = LinkContext {
                level,
                max_connections: healing_ctx.max_connections,
                new_node: healing_ctx.new_node_id,
            };

            if let Some(candidate) = healer.select_new_node_fallback(
                link_ctx,
                healing_ctx.filtered_new_node_neighbours.get(level),
            ) {
                neighbours.push(candidate);
            }
        }
    }

    #[cfg_attr(
        not(debug_assertions),
        expect(dead_code, reason = "test helper unused in release builds")
    )]
    #[cfg(test)]
    pub(crate) fn heal_reachability(&mut self, max_connections: usize) {
        super::test_helpers::TestHelpers::new(self.graph).heal_reachability(max_connections);
    }

    #[cfg(test)]
    pub(crate) fn enforce_bidirectional_all(&mut self, max_connections: usize) {
        super::test_helpers::TestHelpers::new(self.graph)
            .enforce_bidirectional_all(max_connections);
    }
}

#[cfg(test)]
mod tests;
