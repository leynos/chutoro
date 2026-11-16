//! Input generation for MST error-selection property tests.
//!
//! These strategies build candidate-edge lists with several invalid edges at
//! drawn positions, so the lowest-index error contract can be exercised over
//! an open-ended input space. They are kept apart from the graph generators in
//! [`super::strategies`] because they target validation rather than spanning
//! forest construction.

use proptest::prelude::*;

use crate::CandidateEdge;

/// Generates a candidate-edge list containing at least two invalid edges.
///
/// The returned node count and edges exercise lowest-index error selection.
/// Two distinct positions are drawn first, so the vector shrinks towards
/// shorter lists while still carrying two invalid witnesses.
pub(super) fn invalid_edge_list_strategy() -> impl Strategy<Value = (usize, Vec<CandidateEdge>)> {
    (1usize..8).prop_flat_map(|node_count| {
        (2usize..24).prop_flat_map(move |len| {
            (
                distinct_position_pair(len),
                (
                    invalid_edge_strategy(node_count),
                    invalid_edge_strategy(node_count),
                ),
                prop::collection::vec(valid_edge_strategy(node_count), len.saturating_sub(2)),
            )
                .prop_map(move |(positions, invalid_edges, valid_edges)| {
                    (
                        node_count,
                        place_invalid_edges(positions, invalid_edges, valid_edges),
                    )
                })
        })
    })
}

/// Places the two invalid edges at `positions`, filling the other slots with
/// valid edges.
///
/// The generator supplies exactly `len − 2` valid edges, so the three spans
/// cover the list without needing to detect a short input. Drawing the
/// invalid edges as a fixed pair rather than a `Vec` keeps the placement
/// infallible: there is no iterator state to run dry.
fn place_invalid_edges(
    positions: (usize, usize),
    invalid_edges: (CandidateEdge, CandidateEdge),
    valid_edges: Vec<CandidateEdge>,
) -> Vec<CandidateEdge> {
    let (low, high) = ascending_pair(positions);
    let (first_invalid, second_invalid) = invalid_edges;
    let mut valid_iter = valid_edges.into_iter();

    let mut placed = Vec::with_capacity(valid_iter.len().saturating_add(2));
    placed.extend(valid_iter.by_ref().take(low));
    placed.push(first_invalid);
    // The positions are distinct, so at least one valid edge separates them.
    placed.extend(
        valid_iter
            .by_ref()
            .take(high.saturating_sub(low).saturating_sub(1)),
    );
    placed.push(second_invalid);
    placed.extend(valid_iter);
    placed
}

/// Returns the pair with the smaller value first.
fn ascending_pair((first, second): (usize, usize)) -> (usize, usize) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

/// Generates two distinct positions within `0..len`.
fn distinct_position_pair(len: usize) -> impl Strategy<Value = (usize, usize)> {
    (0..len).prop_flat_map(move |first| {
        (
            Just(first),
            (0..len).prop_filter("positions must differ", move |second| *second != first),
        )
    })
}

/// Generates one edge that is invalid for any non-zero node count.
fn invalid_edge_strategy(node_count: usize) -> impl Strategy<Value = CandidateEdge> {
    prop_oneof![
        (0u64..1024).prop_map(move |sequence| CandidateEdge::new(node_count, 0, 1.0, sequence)),
        (0u64..1024).prop_map(move |sequence| CandidateEdge::new(0, node_count, 1.0, sequence)),
        (0u64..1024).prop_map(|sequence| CandidateEdge::new(0, 0, f32::NAN, sequence)),
    ]
}

/// Generates one in-range edge with a finite weight.
fn valid_edge_strategy(node_count: usize) -> impl Strategy<Value = CandidateEdge> {
    (0..node_count, 0..node_count, -100.0f32..100.0, 0u64..1024).prop_map(
        |(source, target, distance, sequence)| {
            CandidateEdge::new(source, target, distance, sequence)
        },
    )
}
