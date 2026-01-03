//! HNSW insertion workflow.
//!
//! Provides the planning and application phases for inserting nodes into the HNSW
//! graph. Planning computes the descent path and layer-by-layer neighbour
//! candidates without holding write locks. Application mutates the graph
//! structure, performs bidirectional linking, and schedules trimming jobs for
//! nodes that exceed maximum connection limits.

mod commit;
mod connectivity;
mod executor;
mod limits;
mod planner;
mod reciprocity;
mod reconciliation;
mod staging;
#[cfg(test)]
mod test_helpers;
mod types;

pub(super) use executor::{InsertionExecutor, TrimJob, TrimResult};
pub(super) use planner::{InsertionPlanner, PlanningInputs};

use crate::hnsw::types::{CandidateEdge, InsertionPlan};

/// Extracts candidate edges from an insertion plan.
///
/// Collects all neighbour relationships discovered during planning as directed
/// edges from the inserted node to each discovered neighbour. Captures edges
/// from all layers to ensure comprehensive MST candidate coverage.
///
/// Self-edges (where `source == target`) are filtered out.
///
/// # Duplicate Edges
///
/// The same neighbour may appear in multiple layers of the insertion plan,
/// resulting in duplicate `(source, target, distance)` tuples with identical
/// sequence numbers. This is intentional behaviour:
///
/// - The downstream MST pipeline treats edges as a simple graph; duplicates
///   are naturally deduplicated when building the union-find structure.
/// - Preserving duplicates here avoids an O(n log n) sort + dedup per insertion
///   in the hot path, deferring any deduplication cost to the final MST phase.
/// - The edge count may exceed the number of unique node pairs, but this does
///   not affect correctness.
///
/// # Arguments
///
/// * `source_node` - The node being inserted
/// * `source_sequence` - Insertion sequence for deterministic tie-breaking
/// * `plan` - The insertion plan containing discovered neighbours per layer
///
/// # Examples
/// ```rust,ignore
/// use crate::hnsw::insert::extract_candidate_edges;
/// use crate::hnsw::types::InsertionPlan;
///
/// let plan = /* ... */;
/// let edges = extract_candidate_edges(5, 42, &plan);
/// assert!(edges.iter().all(|e| e.source() == 5));
/// ```
pub(super) fn extract_candidate_edges(
    source_node: usize,
    source_sequence: u64,
    plan: &InsertionPlan,
) -> Vec<CandidateEdge> {
    plan.layers
        .iter()
        .flat_map(|layer| {
            layer.neighbours.iter().filter_map(move |neighbour| {
                // Filter out self-edges
                if neighbour.id == source_node {
                    return None;
                }
                Some(CandidateEdge::new(
                    source_node,
                    neighbour.id,
                    neighbour.distance,
                    source_sequence,
                ))
            })
        })
        .collect()
}

/// Bundles update parameters for Kani reconciliation helpers.
#[cfg(kani)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct KaniUpdateContext {
    pub(crate) origin: usize,
    pub(crate) level: usize,
    pub(crate) max_connections: usize,
}

#[cfg(kani)]
impl KaniUpdateContext {
    pub(crate) fn new(origin: usize, level: usize, max_connections: usize) -> Self {
        Self {
            origin,
            level,
            max_connections,
        }
    }
}

/// Describes a finalised neighbour update for Kani commit-path harnesses.
#[cfg(kani)]
#[derive(Clone, Debug)]
pub(crate) struct KaniCommitUpdate {
    pub(crate) node: usize,
    pub(crate) level: usize,
    pub(crate) neighbours: Vec<usize>,
}

/// Captures the new-node context for Kani commit-path harnesses.
#[cfg(kani)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct KaniCommitContext {
    pub(crate) new_node: usize,
    pub(crate) new_node_level: usize,
    pub(crate) max_connections: usize,
}

#[cfg(kani)]
impl KaniCommitContext {
    pub(crate) fn new(new_node: usize, new_node_level: usize, max_connections: usize) -> Self {
        Self {
            new_node,
            new_node_level,
            max_connections,
        }
    }
}

#[cfg(kani)]
fn is_deduped(list: &[usize]) -> bool {
    for (idx, value) in list.iter().enumerate() {
        if list[idx + 1..].contains(value) {
            return false;
        }
    }
    true
}

#[cfg(kani)]
fn validate_new_node_for_kani(graph: &crate::hnsw::graph::Graph, ctx: &KaniCommitContext) {
    let new_node_level_valid = graph
        .node(ctx.new_node)
        .map(|node| ctx.new_node_level < node.level_count())
        .unwrap_or(false);
    debug_assert!(
        new_node_level_valid,
        "Kani commit new node must exist and expose the requested level"
    );
    kani::assume(new_node_level_valid);
}

#[cfg(kani)]
fn validate_update_for_kani(
    graph: &crate::hnsw::graph::Graph,
    update: &KaniCommitUpdate,
    max_connections: usize,
) {
    let origin_exists = graph.node(update.node).is_some();
    debug_assert!(
        origin_exists,
        "Kani commit update origin must exist in the graph"
    );
    kani::assume(origin_exists);
    let origin_node = graph
        .node(update.node)
        .expect("Kani commit update origin must exist");
    let origin_level_valid = update.level < origin_node.level_count();
    debug_assert!(
        origin_level_valid,
        "Kani commit update origin must expose the requested level"
    );
    kani::assume(origin_level_valid);

    let deduped = is_deduped(update.neighbours.as_slice());
    debug_assert!(
        deduped,
        "Kani commit update neighbour list must be deduplicated"
    );
    kani::assume(deduped);

    let no_self_loops = !update.neighbours.contains(&update.node);
    debug_assert!(
        no_self_loops,
        "Kani commit update neighbours must not contain the origin"
    );
    kani::assume(no_self_loops);

    let limit = limits::compute_connection_limit(update.level, max_connections);
    let within_limit = update.neighbours.len() <= limit;
    debug_assert!(
        within_limit,
        "Kani commit update neighbours must respect connection limits"
    );
    kani::assume(within_limit);

    let targets_exist = update.neighbours.iter().all(|&id| graph.node(id).is_some());
    debug_assert!(
        targets_exist,
        "Kani commit update neighbours must exist in the graph"
    );
    kani::assume(targets_exist);

    let targets_level_valid = update.neighbours.iter().all(|&id| {
        graph
            .node(id)
            .map(|node| update.level < node.level_count())
            .unwrap_or(false)
    });
    debug_assert!(
        targets_level_valid,
        "Kani commit update neighbours must expose the requested level"
    );
    kani::assume(targets_level_valid);
}

/// Applies the full commit-path update sequence for Kani harnesses.
///
/// This helper drives the same reconciliation and deferred scrub logic used in
/// production by calling [`CommitApplicator::apply_neighbour_updates`] and
/// [`CommitApplicator::apply_new_node_neighbours`]. It constrains inputs to
/// match production preconditions so Kani explores valid states.
#[cfg(kani)]
pub(crate) fn apply_commit_updates_for_kani(
    graph: &mut crate::hnsw::graph::Graph,
    ctx: KaniCommitContext,
    updates: Vec<KaniCommitUpdate>,
) -> Result<(), crate::hnsw::error::HnswError> {
    validate_new_node_for_kani(graph, &ctx);

    for update in &updates {
        validate_update_for_kani(graph, update, ctx.max_connections);
    }

    let final_updates: Vec<types::FinalisedUpdate> = updates
        .into_iter()
        .map(|update| {
            let ctx = crate::hnsw::graph::EdgeContext {
                level: update.level,
                max_connections: ctx.max_connections,
            };
            let staged = types::StagedUpdate {
                node: update.node,
                ctx,
                candidates: update.neighbours.clone(),
            };
            (staged, update.neighbours)
        })
        .collect();

    let new_node = types::NewNodeContext {
        id: ctx.new_node,
        level: ctx.new_node_level,
    };

    let mut applicator = commit::CommitApplicator::new(graph);
    let (reciprocated, _touched) =
        applicator.apply_neighbour_updates(final_updates, ctx.max_connections, new_node)?;
    applicator.apply_new_node_neighbours(ctx.new_node, ctx.new_node_level, reciprocated)?;

    Ok(())
}

/// Applies reconciliation logic to a single update for Kani harnesses.
///
/// This helper mirrors the production commit flow (removed-edge reconciliation,
/// added-edge reconciliation, list write-back, and deferred scrubs) while
/// keeping the setup compact for bounded verification.
///
/// # Examples
/// ```rust,ignore
/// use crate::hnsw::{
///     graph::{Graph, NodeContext},
///     insert::{apply_reconciled_update_for_kani, KaniUpdateContext},
///     params::HnswParams,
/// };
///
/// let params = HnswParams::new(2, 2).expect("params must be valid");
/// let mut graph = Graph::with_capacity(params, 2);
/// graph
///     .insert_first(NodeContext { node: 0, level: 0, sequence: 0 })
///     .expect("insert node 0");
/// graph
///     .attach_node(NodeContext { node: 1, level: 0, sequence: 1 })
///     .expect("attach node 1");
/// let ctx = KaniUpdateContext::new(0, 0, 2);
/// let mut next = vec![1];
/// apply_reconciled_update_for_kani(&mut graph, ctx, &mut next);
/// ```
#[cfg(kani)]
pub(crate) fn apply_reconciled_update_for_kani(
    graph: &mut crate::hnsw::graph::Graph,
    ctx: KaniUpdateContext,
    next: &mut Vec<usize>,
) {
    let previous = {
        let origin_node = graph
            .node(ctx.origin)
            .expect("Kani update origin must exist in the graph");
        let level_valid = ctx.level < origin_node.level_count();
        debug_assert!(
            level_valid,
            "Kani update origin must expose the requested level"
        );
        kani::assume(level_valid);
        let origin_deduped = is_deduped(origin_node.neighbours(ctx.level));
        debug_assert!(
            origin_deduped,
            "Kani update origin neighbours must be deduplicated"
        );
        kani::assume(origin_deduped);
        origin_node.neighbours(ctx.level).to_vec()
    };

    let next_deduped = is_deduped(next);
    debug_assert!(next_deduped, "Kani update next list must be deduplicated");
    kani::assume(next_deduped);
    let next_targets_exist = next.iter().all(|&id| graph.node(id).is_some());
    debug_assert!(
        next_targets_exist,
        "Kani update targets must exist in the graph"
    );
    kani::assume(next_targets_exist);
    let next_targets_level_valid = next.iter().all(|&id| {
        graph
            .node(id)
            .map(|node| ctx.level < node.level_count())
            .unwrap_or(false)
    });
    debug_assert!(
        next_targets_level_valid,
        "Kani update targets must expose the requested level"
    );
    kani::assume(next_targets_level_valid);

    let update_ctx = types::UpdateContext {
        origin: ctx.origin,
        level: ctx.level,
        max_connections: ctx.max_connections,
    };
    let mut reconciler = reconciliation::EdgeReconciler::new(graph);
    reconciler.reconcile_removed_edges(&update_ctx, &previous, next.as_slice());
    reconciler.reconcile_added_edges(&update_ctx, next);

    if let Some(node_ref) = reconciler.graph_mut().node_mut(ctx.origin) {
        let list = node_ref.neighbours_mut(ctx.level);
        list.clear();
        list.extend(next.iter().copied());
    }

    reconciler.apply_deferred_scrubs(ctx.max_connections);
}

/// Ensures a reverse edge using the production reconciler for Kani harnesses.
///
/// This helper calls the same reconciliation code used during insertion
/// commit to ensure a reverse edge exists for a single `(origin, target,
/// level)` tuple.
///
/// # Examples
/// ```rust,ignore
/// use crate::hnsw::{
///     graph::{Graph, NodeContext},
///     insert::{ensure_reverse_edge_for_kani, KaniUpdateContext},
///     params::HnswParams,
/// };
///
/// let params = HnswParams::new(1, 1).expect("params must be valid");
/// let mut graph = Graph::with_capacity(params, 2);
/// graph
///     .insert_first(NodeContext { node: 0, level: 0, sequence: 0 })
///     .expect("insert node 0");
/// graph
///     .attach_node(NodeContext { node: 1, level: 0, sequence: 1 })
///     .expect("attach node 1");
/// let ctx = KaniUpdateContext::new(0, 0, 1);
/// let added = ensure_reverse_edge_for_kani(&mut graph, ctx, 1);
/// assert!(added);
/// ```
#[cfg(kani)]
pub(crate) fn ensure_reverse_edge_for_kani(
    graph: &mut crate::hnsw::graph::Graph,
    ctx: KaniUpdateContext,
    target: usize,
) -> bool {
    {
        let origin_node = graph
            .node(ctx.origin)
            .expect("Kani update origin must exist in the graph");
        let origin_level_valid = ctx.level < origin_node.level_count();
        debug_assert!(
            origin_level_valid,
            "Kani update origin must expose the requested level"
        );
        kani::assume(origin_level_valid);
        let origin_deduped = is_deduped(origin_node.neighbours(ctx.level));
        debug_assert!(
            origin_deduped,
            "Kani update origin neighbours must be deduplicated"
        );
        kani::assume(origin_deduped);
    }
    {
        let target_node = graph
            .node(target)
            .expect("Kani update target must exist in the graph");
        let target_level_valid = ctx.level < target_node.level_count();
        debug_assert!(
            target_level_valid,
            "Kani update target must expose the requested level"
        );
        kani::assume(target_level_valid);
        let target_deduped = is_deduped(target_node.neighbours(ctx.level));
        debug_assert!(
            target_deduped,
            "Kani update target neighbours must be deduplicated"
        );
        kani::assume(target_deduped);
    }

    let update_ctx = types::UpdateContext {
        origin: ctx.origin,
        level: ctx.level,
        max_connections: ctx.max_connections,
    };
    let mut reconciler = reconciliation::EdgeReconciler::new(graph);
    reconciler.ensure_reverse_edge(&update_ctx, target)
}
