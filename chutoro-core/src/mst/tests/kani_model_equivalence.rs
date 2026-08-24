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
    /// Returns the weight this scheme assigns to the edge at `index`.
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
/// Sweeps every edge subset of one-to-four-node complete graphs.
///
/// Each subset runs under every weight scheme and the given deselected
/// edge encoding; model and production must agree exactly.
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
/// Asserts the model rejects invalid inputs with production's errors.
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

/// Exercises the deduplication path, which the complete-graph sweep above
/// cannot reach because it emits each undirected pair at most once.
///
/// Duplicates are in-domain: the model's six-edge budget counts canonical
/// edges before deduplication, so a caller may legitimately supply repeats.
/// Deduplication interacts with the `(weight, source, target, sequence)`
/// ordering, which is where a divergence would hide.
#[rstest]
#[case::identical_repeat(
    3,
    vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 2.0, 1),
    ],
)]
#[case::same_edge_differing_sequence(
    3,
    vec![
        CandidateEdge::new(0, 1, 1.0, 5),
        CandidateEdge::new(0, 1, 1.0, 2),
        CandidateEdge::new(1, 2, 2.0, 1),
    ],
)]
#[case::reversed_orientation_then_duplicate(
    3,
    vec![
        CandidateEdge::new(1, 0, 1.0, 0),
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 2.0, 1),
    ],
)]
#[case::same_pair_differing_weights_must_not_dedupe(
    3,
    vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(0, 1, 3.0, 1),
        CandidateEdge::new(1, 2, 2.0, 2),
    ],
)]
#[case::duplicates_at_the_six_edge_budget(
    4,
    vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 2.0, 1),
        CandidateEdge::new(2, 3, 3.0, 2),
        CandidateEdge::new(0, 3, 4.0, 3),
        CandidateEdge::new(1, 3, 5.0, 4),
    ],
)]
fn model_matches_production_for_duplicate_edges(
    #[case] node_count: usize,
    #[case] candidates: Vec<CandidateEdge>,
) {
    if let Err(divergence) = check_equivalence(node_count, &candidates) {
        panic!("{divergence}");
    }
}

/// Pins the model's bounded domain, where it deliberately diverges.
///
/// Beyond four nodes or six canonical edges the model reports an invariant
/// violation while production succeeds. That is the modelling contract, not
/// a defect: widening a harness past these bounds must fail loudly rather
/// than silently verify a truncated graph.
#[rstest]
#[case::seven_canonical_edges(
    4,
    vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(0, 2, 2.0, 1),
        CandidateEdge::new(0, 3, 3.0, 2),
        CandidateEdge::new(1, 2, 4.0, 3),
        CandidateEdge::new(1, 3, 5.0, 4),
        CandidateEdge::new(2, 3, 6.0, 5),
        CandidateEdge::new(0, 1, 9.0, 6),
    ],
    "Kani MST model supports at most six edges",
)]
#[case::five_nodes(
    5,
    vec![CandidateEdge::new(0, 4, 1.0, 0)],
    "Kani MST model supports at most four nodes",
)]
fn model_rejects_inputs_outside_its_bounded_domain(
    #[case] node_count: usize,
    #[case] candidates: Vec<CandidateEdge>,
    #[case] expected_invariant: &str,
) {
    assert!(
        parallel_kruskal_from_edges(node_count, candidates.iter()).is_ok(),
        "production must accept this input; only the bounded model rejects it",
    );

    let error = kruskal_model(node_count, candidates.iter())
        .expect_err("the model must reject input outside its bounded domain");
    let MstError::InvariantViolation { invariant, .. } = error else {
        panic!("expected an invariant violation, got {error:?}");
    };
    assert_eq!(invariant, expected_invariant);
}
