//! Applies staged HNSW insertions by mutating the graph and scheduling trim
//! jobs.

use std::collections::HashMap;

use crate::hnsw::{
    error::HnswError,
    graph::{ApplyContext, Graph, NodeContext},
    params::connection_limit_for_level,
};

use super::commit::CommitApplicator;
use super::connectivity::ConnectivityHealer;
use super::reciprocity::{ReciprocityEnforcer, ReciprocityWorkspace};
use super::staging::InsertionStager;
use super::types::{FinalisedUpdate, LinkContext, NewNodeContext, PreparedInsertion, TrimWork};

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
    #[expect(
        clippy::excessive_nesting,
        reason = "Fallback selection and connectivity healing require structured branching"
    )]
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
            &filtered_new_node_neighbours,
            new_node.id,
            max_connections,
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
        filtered_new_node_neighbours: &[Vec<usize>],
        new_node_id: usize,
        max_connections: usize,
    ) {
        let mut healer = ConnectivityHealer::new(self.graph);
        for (level, neighbours) in reciprocated.iter_mut().enumerate() {
            neighbours.sort_unstable();
            neighbours.dedup();
            let limit = compute_connection_limit(level, max_connections);
            if neighbours.len() > limit {
                neighbours.truncate(limit);
            }
            if !neighbours.is_empty() {
                continue;
            }

            let link_ctx = LinkContext {
                level,
                max_connections,
                new_node: new_node_id,
            };

            if let Some(candidate) = healer.select_new_node_fallback(
                link_ctx,
                filtered_new_node_neighbours.get(level),
            ) {
                neighbours.push(candidate);
            }
        }
    }

    #[cfg_attr(not(debug_assertions), allow(dead_code))]
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

/// Computes the connection limit for a given level (doubled for level 0).
pub(super) fn compute_connection_limit(level: usize, max_connections: usize) -> usize {
    connection_limit_for_level(level, max_connections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::insert::{reconciliation::EdgeReconciler, test_helpers::TestHelpers, types};
    use crate::hnsw::{
        graph::{Graph, NodeContext},
        params::HnswParams,
    };

    #[test]
    fn ensure_reverse_edge_evicts_and_scrubs_forward_link() {
        let params = HnswParams::new(1, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 3);

        graph
            .insert_first(NodeContext {
                node: 0,
                level: 1,
                sequence: 0,
            })
            .expect("insert entry");
        graph
            .attach_node(NodeContext {
                node: 1,
                level: 1,
                sequence: 1,
            })
            .expect("attach node 1");
        graph
            .attach_node(NodeContext {
                node: 2,
                level: 1,
                sequence: 2,
            })
            .expect("attach node 2");

        // Forward edges: 0 -> 1, 2 -> 1; target (1) is at capacity and prefers 2.
        graph.node_mut(0).unwrap().neighbours_mut(1).push(1);
        graph.node_mut(1).unwrap().neighbours_mut(1).push(2);
        graph.node_mut(2).unwrap().neighbours_mut(1).push(1);

        let mut reconciler = EdgeReconciler::new(&mut graph);
        let ensured = reconciler.ensure_reverse_edge(
            &types::UpdateContext {
                origin: 0,
                level: 1,
                max_connections: 1,
            },
            1,
        );

        assert!(ensured, "reverse edge should be ensured even when evicting");

        let target = reconciler.graph.node(1).unwrap();
        assert_eq!(target.neighbours(1), &[0]);

        let evicted = reconciler.graph.node(2).unwrap();
        assert!(
            !evicted.neighbours(1).contains(&1),
            "evicted neighbour should lose its forward edge to maintain reciprocity",
        );

        let origin = reconciler.graph.node(0).unwrap();
        assert!(origin.neighbours(1).contains(&1));
    }

    #[test]
    fn ensure_new_node_reciprocity_removes_one_way_edges() {
        let params = HnswParams::new(1, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 2);

        graph
            .insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })
            .expect("insert entry");
        graph
            .attach_node(NodeContext {
                node: 1,
                level: 0,
                sequence: 1,
            })
            .expect("attach node 1");

        graph.node_mut(1).unwrap().neighbours_mut(0).push(0);

        let mut enforcer = ReciprocityEnforcer::new(&mut graph);
        enforcer.ensure_reciprocity_for_touched(&[(1, 0)], 1);

        let node0 = enforcer.graph.node(0).unwrap();
        let node1 = enforcer.graph.node(1).unwrap();

        assert!(node0.neighbours(0).contains(&1));
        assert!(node1.neighbours(0).contains(&0));
    }

    #[test]
    fn ensure_reciprocity_for_touched_heals_existing_one_way() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 3);

        graph
            .insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })
            .expect("insert entry");
        graph
            .attach_node(NodeContext {
                node: 1,
                level: 0,
                sequence: 1,
            })
            .expect("attach node 1");
        graph
            .attach_node(NodeContext {
                node: 2,
                level: 0,
                sequence: 2,
            })
            .expect("attach node 2");

        // One-way edge from node 2 to node 0.
        graph.node_mut(2).unwrap().neighbours_mut(0).push(0);

        let mut enforcer = ReciprocityEnforcer::new(&mut graph);
        enforcer.ensure_reciprocity_for_touched(&[(2, 0)], 2);

        let node0 = enforcer.graph.node(0).unwrap();
        let node2 = enforcer.graph.node(2).unwrap();

        assert!(node0.neighbours(0).contains(&2));
        assert!(node2.neighbours(0).contains(&0));
    }

    #[test]
    fn enforce_bidirectional_all_adds_upper_layer_backlink() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 2);

        graph
            .insert_first(NodeContext {
                node: 0,
                level: 1,
                sequence: 0,
            })
            .expect("insert entry");
        graph
            .attach_node(NodeContext {
                node: 1,
                level: 1,
                sequence: 1,
            })
            .expect("attach node 1");

        graph.node_mut(0).unwrap().neighbours_mut(1).push(1);

        TestHelpers::new(&mut graph).enforce_bidirectional_all(2);

        let node0 = graph.node(0).unwrap();
        let node1 = graph.node(1).unwrap();

        assert!(node0.neighbours(1).contains(&1));
        assert!(node1.neighbours(1).contains(&0));
    }

    #[test]
    fn enforce_bidirectional_all_removes_invalid_upper_edge() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 2);

        graph
            .insert_first(NodeContext {
                node: 0,
                level: 1,
                sequence: 0,
            })
            .expect("insert entry");
        graph
            .attach_node(NodeContext {
                node: 1,
                level: 0,
                sequence: 1,
            })
            .expect("attach node 1");

        // One-way edge exists at level 1, but target only has level 0.
        graph.node_mut(0).unwrap().neighbours_mut(1).push(1);

        TestHelpers::new(&mut graph).enforce_bidirectional_all(2);

        let node0 = graph.node(0).unwrap();
        assert!(node0.neighbours(1).is_empty());
    }
}
