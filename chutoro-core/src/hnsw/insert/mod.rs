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
///     insert::apply_reconciled_update_for_kani,
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
/// let mut next = vec![1];
/// apply_reconciled_update_for_kani(&mut graph, 0, 0, 2, &mut next);
/// ```
#[cfg(kani)]
pub(crate) fn apply_reconciled_update_for_kani(
    graph: &mut crate::hnsw::graph::Graph,
    origin: usize,
    level: usize,
    max_connections: usize,
    next: &mut Vec<usize>,
) {
    let previous = graph
        .node(origin)
        .map(|node| node.neighbours(level).to_vec())
        .unwrap_or_default();

    let ctx = types::UpdateContext {
        origin,
        level,
        max_connections,
    };
    let mut reconciler = reconciliation::EdgeReconciler::new(graph);
    let next_snapshot = next.clone();
    reconciler.reconcile_removed_edges(&ctx, &previous, &next_snapshot);
    reconciler.reconcile_added_edges(&ctx, next);

    if let Some(node_ref) = reconciler.graph_mut().node_mut(origin) {
        let list = node_ref.neighbours_mut(level);
        list.clear();
        list.extend(next.iter().copied());
    }

    reconciler.apply_deferred_scrubs(max_connections);
}
