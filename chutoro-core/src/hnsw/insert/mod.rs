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
