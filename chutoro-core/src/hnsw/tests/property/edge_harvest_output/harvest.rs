//! Harvest algorithm used by the candidate edge harvest output suite.
//!
//! The harvested output is defined as the union of:
//! - Mutual top-k neighbour edges (k derived from topology metadata).
//! - Minimum spanning forest edges from the input graph (to preserve connectivity).

use std::collections::{BTreeMap, BTreeSet};

use crate::{CandidateEdge, EdgeHarvest, parallel_kruskal};

use super::super::graph_metrics::{degree_ceiling_for_metadata, top_k_neighbour_sets};
use super::super::types::{GraphFixture, GraphMetadata};

/// Picks the top-k value used when harvesting edges for a topology.
pub(super) fn harvest_k_for_metadata(metadata: &GraphMetadata) -> usize {
    let ceiling = degree_ceiling_for_metadata(metadata).max(1);
    let max_k = match metadata {
        GraphMetadata::Disconnected { .. } => 2,
        _ => 5,
    };
    let baseline = ceiling.saturating_sub(1).max(2);
    baseline.min(max_k).min(ceiling)
}

fn canonical_pair(left: usize, right: usize) -> (usize, usize) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn build_edge_lookup(edges: &[CandidateEdge]) -> BTreeMap<(usize, usize), CandidateEdge> {
    let mut lookup = BTreeMap::new();
    for edge in edges {
        let canonical = edge.canonicalise();
        lookup
            .entry((canonical.source(), canonical.target()))
            .or_insert(canonical);
    }
    lookup
}

/// Harvests candidate edges from the provided graph fixture.
pub(super) fn harvest_candidate_edges(
    fixture: &GraphFixture,
) -> Result<Vec<CandidateEdge>, String> {
    let node_count = fixture.graph.node_count;
    let edges = &fixture.graph.edges;
    if node_count <= 1 || edges.is_empty() {
        return Ok(Vec::new());
    }

    let k = harvest_k_for_metadata(&fixture.graph.metadata).min(node_count.saturating_sub(1));
    let top_k = top_k_neighbour_sets(node_count, edges, k);

    let mut selected_pairs = BTreeSet::new();
    for (node, neighbours) in top_k.iter().enumerate() {
        for &neighbour in neighbours {
            if top_k[neighbour].contains(&node) {
                selected_pairs.insert(canonical_pair(node, neighbour));
            }
        }
    }

    let harvest = EdgeHarvest::new(edges.clone());
    let forest = parallel_kruskal(node_count, &harvest)
        .map_err(|err| format!("harvest MST failed: {err}"))?;
    for edge in forest.edges() {
        selected_pairs.insert((edge.source(), edge.target()));
    }

    let lookup = build_edge_lookup(edges);
    let mut output = Vec::with_capacity(selected_pairs.len());
    for (source, target) in selected_pairs {
        let edge = lookup.get(&(source, target)).ok_or_else(|| {
            format!("harvest lookup missing edge ({source}, {target}) for {node_count} nodes")
        })?;
        output.push(*edge);
    }

    Ok(output)
}
