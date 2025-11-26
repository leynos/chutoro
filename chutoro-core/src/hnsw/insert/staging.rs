//! Stages neighbour updates for an insertion before trimming is applied.
//!
//! This module gathers candidate neighbour lists for the new node and affected
//! existing nodes, recording which lists will require trimming once distances
//! are known. It intentionally leaves trimming decisions to callers so that
//! distance computation can occur without holding graph locks.

use std::collections::{HashMap, HashSet};

use crate::hnsw::{
    error::HnswError,
    graph::{EdgeContext, Graph, NodeContext},
    types::InsertionPlan,
};

use super::limits::compute_connection_limit;
use super::types::{LayerProcessingOutcome, StagedUpdate, TrimJob, TrimWork};

#[derive(Debug)]
pub(super) struct InsertionStager<'graph> {
    pub(super) graph: &'graph Graph,
}

impl<'graph> InsertionStager<'graph> {
    pub(super) fn new(graph: &'graph Graph) -> Self {
        Self { graph }
    }

    pub(super) fn ensure_slot_available(&self, node: usize) -> Result<(), HnswError> {
        if !self.graph.has_slot(node) {
            return Err(HnswError::InvalidParameters {
                reason: format!("node {node} is outside pre-allocated capacity"),
            });
        }
        if self.graph.node(node).is_some() {
            return Err(HnswError::DuplicateNode { node });
        }
        Ok(())
    }

    /// Processes the insertion layers, staging neighbour lists and identifying
    /// nodes that will require trimming once distances are available.
    ///
    /// The provided [`NodeContext`] identifies the new node and the highest
    /// level that should be considered during staging.
    pub(super) fn process_insertion_layers(
        &self,
        ctx: NodeContext,
        plan: InsertionPlan,
        max_connections: usize,
    ) -> Result<LayerProcessingOutcome, HnswError> {
        let mut new_node_neighbours = vec![Vec::new(); ctx.level + 1];
        let mut staged: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut initialised = HashSet::new();
        let mut needs_trim = HashSet::new();

        for layer in plan
            .layers
            .into_iter()
            .filter(|layer| layer.level <= ctx.level)
        {
            let level_index = layer.level;
            let level_capacity = compute_connection_limit(level_index, max_connections);

            for neighbour in layer.neighbours.into_iter().take(level_capacity) {
                self.stage_neighbour(
                    ctx.node,
                    neighbour.id,
                    level_index,
                    level_capacity,
                    &mut new_node_neighbours,
                    &mut staged,
                    &mut initialised,
                    &mut needs_trim,
                )?;
            }
        }

        Ok((new_node_neighbours, staged, initialised, needs_trim))
    }

    /// Builds staged updates and trimming jobs from the collected neighbour
    /// candidates.
    pub(super) fn generate_updates_and_trim_jobs(
        &self,
        new_node: NodeContext,
        work: TrimWork,
    ) -> Result<(Vec<StagedUpdate>, Vec<TrimJob>), HnswError> {
        let TrimWork {
            mut staged,
            needs_trim,
            max_connections,
        } = work;
        let mut updates = Vec::with_capacity(staged.len());
        let mut trim_jobs = Vec::with_capacity(needs_trim.len());

        for ((other, lvl), mut candidates) in staged.drain() {
            Self::dedupe_candidates(&mut candidates);
            let ctx = EdgeContext {
                level: lvl,
                max_connections,
            };
            prioritise_new_node(new_node.node, &mut candidates);
            let mut sequences = Vec::with_capacity(candidates.len());
            for &candidate in &candidates {
                sequences.push(self.sequence_for_candidate(candidate, new_node, lvl)?);
            }
            if needs_trim.contains(&(other, lvl)) {
                let reordered = candidates.clone();
                debug_assert_eq!(
                    reordered.len(),
                    sequences.len(),
                    "trim job sequences must align with candidates",
                );
                trim_jobs.push(TrimJob {
                    node: other,
                    ctx,
                    candidates: reordered,
                    sequences,
                });
            }
            updates.push(StagedUpdate {
                node: other,
                ctx,
                candidates,
            });
        }

        Ok((updates, trim_jobs))
    }

    pub(super) fn dedupe_new_node_lists(levels: &mut [Vec<usize>]) {
        for neighbours in levels {
            let mut seen = HashSet::new();
            neighbours.retain(|neighbour| seen.insert(*neighbour));
        }
    }

    pub(super) fn dedupe_candidates(candidates: &mut Vec<usize>) {
        candidates.sort_unstable();
        candidates.dedup();
    }

    pub(super) fn sequence_for_candidate(
        &self,
        candidate: usize,
        new_node: NodeContext,
        level: usize,
    ) -> Result<u64, HnswError> {
        if candidate == new_node.node {
            return Ok(new_node.sequence);
        }
        self.graph
            .node_sequence(candidate)
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: format!(
                    "insertion planning: sequence missing for node {candidate} at level {level}",
                ),
            })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Staging shares tightly-coupled accumulators; refactoring into a tracker is follow-up work"
    )]
    pub(super) fn stage_neighbour(
        &self,
        new_node: usize,
        neighbour: usize,
        level_index: usize,
        connection_limit: usize,
        new_node_neighbours: &mut [Vec<usize>],
        staged: &mut HashMap<(usize, usize), Vec<usize>>,
        initialised: &mut HashSet<(usize, usize)>,
        needs_trim: &mut HashSet<(usize, usize)>,
    ) -> Result<(), HnswError> {
        new_node_neighbours[level_index].push(neighbour);

        let key = (neighbour, level_index);
        if initialised.insert(key) {
            let graph_node =
                self.graph
                    .node(neighbour)
                    .ok_or_else(|| HnswError::GraphInvariantViolation {
                        message: format!(
                            "insertion planning: node {neighbour} missing at level {level_index}",
                        ),
                    })?;
            staged.insert(key, graph_node.neighbours(level_index).to_vec());
        }

        let candidates =
            staged
                .get_mut(&key)
                .ok_or_else(|| HnswError::GraphInvariantViolation {
                    message: format!(
                        "insertion planning: node {neighbour} missing from staged updates at level {level_index}",
                    ),
                })?;

        let contains_new = candidates.contains(&new_node);
        let projected = candidates.len() + usize::from(!contains_new);
        if projected > connection_limit {
            needs_trim.insert(key);
        }
        if contains_new {
            return Ok(());
        }

        candidates.push(new_node);
        Ok(())
    }
}

#[inline]
pub(super) fn prioritise_new_node(new_node: usize, candidates: &mut [usize]) {
    if let Some(pos) = candidates
        .iter()
        .position(|&candidate| candidate == new_node)
    {
        if pos != 0 {
            candidates.swap(0, pos);
        }
    }
}
