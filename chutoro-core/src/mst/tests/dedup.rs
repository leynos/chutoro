//! Regression tests for duplicate removal in edge preparation.
//!
//! The forest alone cannot prove that preparation deduplicates: a second copy
//! of an accepted edge would be rejected by the union-find as a cycle, so the
//! resulting edge list is identical either way. These tests call
//! `prepare_edge_list` directly and inspect the list it returns, which is the
//! only place the preparation-time contract is observable.
//!
//! The contract under test, from `prepare_edge_list`'s sort and
//! `dedup_by` pass: copies that canonicalize to the same weight and endpoints
//! collapse to one record, and the survivor is the copy with the lowest
//! `sequence`, because the sort orders by `sequence` last and `dedup_by` keeps
//! the first of each run.

use crate::CandidateEdge;

use super::super::prepare_edge_list;

/// Builds candidate edges from `(source, target, weight, sequence)` tuples.
fn edges(tuples: &[(usize, usize, f32, u64)]) -> Vec<CandidateEdge> {
    tuples
        .iter()
        .map(|(source, target, weight, sequence)| {
            CandidateEdge::new(*source, *target, *weight, *sequence)
        })
        .collect()
}

/// Returns `(source, target, weight bits, sequence)` for every prepared edge.
fn identities(prepared: &[super::super::MstEdge]) -> Vec<(usize, usize, u32, u64)> {
    prepared
        .iter()
        .map(|edge| {
            (
                edge.source(),
                edge.target(),
                edge.weight().to_bits(),
                edge.sequence(),
            )
        })
        .collect()
}

#[test]
fn equal_weight_duplicates_keep_the_lowest_sequence() {
    // Four copies of the same undirected edge, supplied out of sequence order
    // so the survivor is chosen by the sort and not by input position.
    let prepared = prepare_edge_list(
        &edges(&[
            (0, 1, 2.5, 30),
            (0, 1, 2.5, 10),
            (0, 1, 2.5, 40),
            (0, 1, 2.5, 20),
        ]),
        2,
    )
    .expect("a valid graph must prepare");

    assert_eq!(
        identities(&prepared),
        vec![(0, 1, 2.5_f32.to_bits(), 10)],
        "the lowest-sequence duplicate must be the only survivor"
    );
}

#[test]
fn reversed_orientation_canonicalizes_before_deduplication() {
    // `(1, 0)` and `(0, 1)` are the same undirected edge, so they collapse;
    // the lower sequence survives regardless of which orientation carries it.
    let prepared = prepare_edge_list(&edges(&[(1, 0, 4.0, 7), (0, 1, 4.0, 3)]), 2)
        .expect("a valid graph must prepare");

    assert_eq!(
        identities(&prepared),
        vec![(0, 1, 4.0_f32.to_bits(), 3)],
        "opposite orientations of one edge must collapse to its lowest sequence"
    );
}

#[test]
fn same_endpoints_with_different_weights_are_both_kept() {
    // The dedup key includes the weight, so parallel edges between the same
    // pair with different weights are distinct candidates and must survive.
    let prepared = prepare_edge_list(&edges(&[(0, 1, 2.0, 1), (0, 1, 3.0, 2)]), 2)
        .expect("a valid graph must prepare");

    assert_eq!(
        identities(&prepared),
        vec![(0, 1, 2.0_f32.to_bits(), 1), (0, 1, 3.0_f32.to_bits(), 2),],
        "different weights must not be deduplicated"
    );
}

#[test]
fn self_loops_are_removed_before_deduplication() {
    // A self-loop is dropped by validation, so it can neither survive nor
    // shadow a usable edge that shares its sequence range.
    let prepared = prepare_edge_list(&edges(&[(0, 0, 1.0, 1), (0, 1, 2.0, 2)]), 2)
        .expect("a valid graph must prepare");

    assert_eq!(
        identities(&prepared),
        vec![(0, 1, 2.0_f32.to_bits(), 2)],
        "the self-loop must be dropped and the usable edge kept"
    );
}
