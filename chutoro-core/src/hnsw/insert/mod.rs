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
#[cfg(any(test, kani))]
pub(crate) mod test_helpers;
mod types;

pub(super) use executor::{InsertionExecutor, TrimJob, TrimResult};
pub(super) use planner::{InsertionPlanner, PlanningInputs};
#[cfg(kani)]
pub(crate) use types::{FinalisedUpdate, NewNodeContext, StagedUpdate};

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

/// Checks whether a slice contains no duplicate elements.
///
/// Returns `true` if every element appears exactly once.
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
/// Assumes the node exists, exposes the requested level, and has a deduplicated
/// neighbour list at that level.
fn assume_node_has_level(
    graph: &crate::hnsw::graph::Graph,
    node_id: usize,
    level: usize,
    msg: &'static str,
) {
    let node = graph.node(node_id);
    let node_exists = node.is_some();
    debug_assert!(node_exists, "{msg}: node must exist");
    kani::assume(node_exists);

    let level_valid = node.map(|node| level < node.level_count()).unwrap_or(false);
    debug_assert!(level_valid, "{msg}: node must expose the requested level");
    kani::assume(level_valid);

    let neighbours_deduped = node
        .map(|node| is_deduped(node.neighbours(level)))
        .unwrap_or(false);
    debug_assert!(neighbours_deduped, "{msg}: neighbours must be deduplicated");
    kani::assume(neighbours_deduped);
}

#[cfg(kani)]
fn validate_new_node_for_kani(graph: &crate::hnsw::graph::Graph, new_node: &types::NewNodeContext) {
    assume_node_has_level(graph, new_node.id, new_node.level, "Kani commit new node");
}

#[cfg(kani)]
fn validate_update_for_kani(
    graph: &crate::hnsw::graph::Graph,
    update: &types::FinalisedUpdate,
    max_connections: usize,
) {
    let (staged, neighbours) = update;
    assume_node_has_level(
        graph,
        staged.node,
        staged.ctx.level,
        "Kani commit update origin",
    );

    let deduped = is_deduped(neighbours.as_slice());
    debug_assert!(
        deduped,
        "Kani commit update neighbour list must be deduplicated"
    );
    kani::assume(deduped);

    let no_self_loops = !neighbours.contains(&staged.node);
    debug_assert!(
        no_self_loops,
        "Kani commit update neighbours must not contain the origin"
    );
    kani::assume(no_self_loops);

    let limit = limits::compute_connection_limit(staged.ctx.level, max_connections);
    let within_limit = neighbours.len() <= limit;
    debug_assert!(
        within_limit,
        "Kani commit update neighbours must respect connection limits"
    );
    kani::assume(within_limit);

    let targets_exist = neighbours.iter().all(|&id| graph.node(id).is_some());
    debug_assert!(
        targets_exist,
        "Kani commit update neighbours must exist in the graph"
    );
    kani::assume(targets_exist);

    let targets_level_valid = neighbours.iter().all(|&id| {
        graph
            .node(id)
            .map(|node| staged.ctx.level < node.level_count())
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
    max_connections: usize,
    new_node: types::NewNodeContext,
    updates: Vec<types::FinalisedUpdate>,
) -> Result<(), crate::hnsw::error::HnswError> {
    validate_new_node_for_kani(graph, &new_node);

    for update in &updates {
        validate_update_for_kani(graph, update, max_connections);
    }

    let mut applicator = commit::CommitApplicator::new(graph);
    let (reciprocated, _touched) =
        applicator.apply_neighbour_updates(updates, max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

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
    assume_node_has_level(graph, ctx.origin, ctx.level, "Kani update origin");
    let previous = graph
        .node(ctx.origin)
        .expect("Kani update origin must exist in the graph")
        .neighbours(ctx.level)
        .to_vec();

    let next_deduped = is_deduped(next);
    debug_assert!(next_deduped, "Kani update next list must be deduplicated");
    kani::assume(next_deduped);
    for &target in next.iter() {
        assume_node_has_level(graph, target, ctx.level, "Kani update target");
    }

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
    assume_node_has_level(graph, ctx.origin, ctx.level, "Kani update origin");
    assume_node_has_level(graph, target, ctx.level, "Kani update target");

    let update_ctx = types::UpdateContext {
        origin: ctx.origin,
        level: ctx.level,
        max_connections: ctx.max_connections,
    };
    let mut reconciler = reconciliation::EdgeReconciler::new(graph);
    reconciler.ensure_reverse_edge(&update_ctx, target)
}
