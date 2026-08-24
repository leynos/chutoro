//! Exhaustive equivalence tests for the sequential Kani MST model.
//!
//! The Kani harnesses verify `kani_model::kruskal_model` rather than the
//! Rayon-based production path, so the model is a modelling boundary. These
//! tests close that boundary by enumerating every edge subset of the complete
//! graph on one to four nodes, under several weight assignments and both
//! subset encodings used by the harnesses, and asserting that the model and
//! the production implementation produce identical forests.

use rstest::rstest;

use crate::CandidateEdge;

use super::super::kani_model::kruskal_model;
use super::super::{MstError, parallel_kruskal_from_edges};

/// How deselected complete-graph edges are represented in the input.
#[derive(Clone, Copy, Debug)]
enum DeselectedEdge {
    /// The edge is omitted from the input entirely.
    Omitted,
    /// The edge degenerates to a self-loop, mirroring the harness encoding.
    SelfLoop,
}

/// Weight assignment applied to the complete-graph edge at `index`.
#[derive(Clone, Copy, Debug)]
enum WeightScheme {
    AllEqual,
    Ascending,
    Descending,
    PairedDuplicates,
}

impl WeightScheme {
    fn weight(self, index: usize, edge_total: usize) -> f32 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "edge indices are at most six, far below f32 precision limits"
        )]
        match self {
            Self::AllEqual => 1.0,
            Self::Ascending => index as f32,
            Self::Descending => (edge_total - index) as f32,
            Self::PairedDuplicates => (index / 2) as f32,
        }
    }
}

/// Returns the complete-graph edge list for `node_count` nodes.
fn complete_graph_pairs(node_count: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for source in 0..node_count {
        for target in (source + 1)..node_count {
            pairs.push((source, target));
        }
    }
    pairs
}

/// Builds the candidate edges for one subset mask of the complete graph.
fn build_candidates(
    pairs: &[(usize, usize)],
    mask: usize,
    scheme: WeightScheme,
    deselected: DeselectedEdge,
) -> Vec<CandidateEdge> {
    let mut candidates = Vec::new();
    for (index, &(source, target)) in pairs.iter().enumerate() {
        let selected = mask & (1 << index) != 0;
        let weight = scheme.weight(index, pairs.len());
        let sequence = index as u64;
        if selected {
            candidates.push(CandidateEdge::new(source, target, weight, sequence));
        } else if matches!(deselected, DeselectedEdge::SelfLoop) {
            candidates.push(CandidateEdge::new(source, source, weight, sequence));
        }
    }
    candidates
}

/// Compares the model and production forests for one input, returning a
/// description of the first divergence.
fn check_equivalence(node_count: usize, candidates: &[CandidateEdge]) -> Result<(), String> {
    let production = parallel_kruskal_from_edges(node_count, candidates.iter())
        .map_err(|error| format!("production Kruskal failed: {error}"))?;
    let model = kruskal_model(node_count, candidates.iter())
        .map_err(|error| format!("Kani model failed: {error}"))?;

    if model.edges() != production.edges() {
        return Err(format!(
            "edge mismatch for nodes={node_count}, input={candidates:?}: \
             model={:?}, production={:?}",
            model.edges(),
            production.edges(),
        ));
    }
    if model.component_count() != production.component_count() {
        return Err(format!(
            "component mismatch for nodes={node_count}, input={candidates:?}: \
             model={}, production={}",
            model.component_count(),
            production.component_count(),
        ));
    }
    Ok(())
}

#[rstest]
#[case::omitted(DeselectedEdge::Omitted)]
#[case::self_loop(DeselectedEdge::SelfLoop)]
fn model_matches_production_for_all_bounded_graphs(#[case] deselected: DeselectedEdge) {
    let schemes = [
        WeightScheme::AllEqual,
        WeightScheme::Ascending,
        WeightScheme::Descending,
        WeightScheme::PairedDuplicates,
    ];

    for node_count in 1..=4 {
        let pairs = complete_graph_pairs(node_count);
        for scheme in schemes {
            for mask in 0..(1usize << pairs.len()) {
                let candidates = build_candidates(&pairs, mask, scheme, deselected);
                if let Err(divergence) = check_equivalence(node_count, &candidates) {
                    panic!("{divergence}");
                }
            }
        }
    }
}

#[rstest]
#[case::empty_graph(0, vec![], MstError::EmptyGraph)]
#[case::invalid_source(
    2,
    vec![CandidateEdge::new(5, 1, 1.0, 0)],
    MstError::InvalidNodeId { node: 5, node_count: 2 },
)]
#[case::invalid_target(
    2,
    vec![CandidateEdge::new(0, 7, 1.0, 0)],
    MstError::InvalidNodeId { node: 7, node_count: 2 },
)]
#[case::non_finite_weight(
    2,
    vec![CandidateEdge::new(0, 1, f32::NAN, 0)],
    MstError::NonFiniteWeight { left: 0, right: 1 },
)]
fn model_matches_production_errors(
    #[case] node_count: usize,
    #[case] candidates: Vec<CandidateEdge>,
    #[case] expected: MstError,
) {
    let production = parallel_kruskal_from_edges(node_count, candidates.iter());
    let model = kruskal_model(node_count, candidates.iter());

    assert_eq!(
        production.expect_err("production must reject input"),
        expected
    );
    assert_eq!(model.expect_err("model must reject input"), expected);
}
