//! Duplicate-heavy fixtures for the exact-edge-list oracle property.
//!
//! The generated distributions repeat weights but do not deliberately repeat
//! one canonical edge with different sequence numbers, so the oracle property
//! never exercises preparation's dedup tie-break on its own. These fixtures
//! supply that shape directly, so a regression in either the sort order or the
//! surviving copy is visible as an edge-list mismatch.
//!
//! The forest alone cannot discriminate here: a second copy of an accepted
//! edge is cycle-rejected by the union-find, so the accepted list is identical
//! whether preparation deduplicated or not. `mst::tests::dedup` therefore
//! inspects `prepare_edge_list` directly; this module complements it at the
//! property level.

use rstest::rstest;

use crate::CandidateEdge;

use super::equivalence::run_oracle_equivalence_property;
use super::types::{MstFixture, WeightDistribution};

/// Builds a fixture whose edges are dominated by equal-weight duplicates.
///
/// For every adjacent node pair it supplies one weight carried by three
/// differently oriented and ordered copies, plus a second distinct weight on
/// the same pair. For example, the `0`-`1` pair appears as `(0, 1, 2.0, 3)`,
/// `(1, 0, 2.0, 7)` and `(0, 1, 2.0, 11)`, which must collapse to the
/// `sequence == 3` record, alongside `(0, 1, 5.0, 19)`, which must survive
/// separately.
fn duplicate_heavy_fixture(node_count: usize) -> MstFixture {
    let mut edges = Vec::new();
    let mut sequence = 0_u64;
    for pair in 0..node_count.saturating_sub(1) {
        let (left, right) = (pair, pair + 1);
        edges.push(CandidateEdge::new(left, right, 2.0, sequence));
        edges.push(CandidateEdge::new(right, left, 2.0, sequence + 4));
        edges.push(CandidateEdge::new(left, right, 2.0, sequence + 8));
        edges.push(CandidateEdge::new(left, right, 5.0, sequence + 16));
        sequence += 20;
    }
    MstFixture {
        node_count,
        edges,
        distribution: WeightDistribution::ManyIdentical,
    }
}

#[rstest]
#[case::two_nodes(2)]
#[case::three_nodes(3)]
#[case::eight_nodes(8)]
#[case::thirty_three_nodes(33)]
fn oracle_equivalence_holds_for_duplicate_heavy_graphs(#[case] node_count: usize) {
    run_oracle_equivalence_property(&duplicate_heavy_fixture(node_count))
        .expect("duplicate-heavy graphs must match the oracle exactly");
}
