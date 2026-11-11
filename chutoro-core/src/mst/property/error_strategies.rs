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
                prop::collection::vec(invalid_edge_strategy(node_count), 2),
                prop::collection::vec(valid_edge_strategy(node_count), len.saturating_sub(2)),
            )
                .prop_map(move |(positions, invalid_edges, valid_edges)| {
                    (
                        node_count,
                        place_invalid_edges(len, positions, invalid_edges, valid_edges),
                    )
                })
        })
    })
}

/// Places the two invalid edges at `positions`, filling the other slots with
/// valid edges.
fn place_invalid_edges(
    len: usize,
    positions: (usize, usize),
    invalid_edges: Vec<CandidateEdge>,
    valid_edges: Vec<CandidateEdge>,
) -> Vec<CandidateEdge> {
    let (low, high) = ascending_pair(positions);
    let mut invalid_iter = invalid_edges.into_iter();
    let mut valid_iter = valid_edges.into_iter();
    (0..len)
        .map(|index| {
            if index == low || index == high {
                invalid_iter.next().expect("exactly two invalid edges")
            } else {
                valid_iter.next().expect("enough valid edges")
            }
        })
        .collect()
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
