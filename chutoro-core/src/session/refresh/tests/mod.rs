//! Unit tests for the pure refresh rebuild.
//!
//! Every case here drives [`super::rebuild_mst_labels`] directly: no HNSW
//! index, no session, no clock. The fixtures are hand-built raw edge lists so
//! the expected labels and backbone are derivable by reading the case.

use std::num::NonZeroUsize;

use googletest::prelude::*;
use pretty_assertions::assert_eq;
use rstest::rstest;

mod recovery;
mod warm_up;

use super::{RefreshInputs, rebuild_mst_labels};
use crate::{CandidateEdge, ChutoroError, MstError};

/// Cluster-size threshold of two, the smallest that forms a cluster here.
///
/// The `None` arm is rejected during constant evaluation, so a zero literal is
/// a compile error rather than a runtime panic inside a fixture.
const MIN_CLUSTER_SIZE_TWO: NonZeroUsize = match NonZeroUsize::new(2) {
    Some(size) => size,
    None => panic!("a test cluster size must be non-zero"),
};

/// Warm-up threshold above the three- and four-point fixtures.
const MIN_CLUSTER_SIZE_FIVE: NonZeroUsize = match NonZeroUsize::new(5) {
    Some(size) => size,
    None => panic!("a test cluster size must be non-zero"),
};

/// Builds inputs with no retained edges, so a case states only its pending set.
fn inputs<'a>(
    node_count: usize,
    pending_edges: &'a [CandidateEdge],
    core_distances: &'a [f32],
    min_size: NonZeroUsize,
) -> RefreshInputs<'a> {
    RefreshInputs {
        node_count,
        mst_backbone: &[],
        historical_edges: &[],
        pending_edges,
        core_distances,
        min_cluster_size: min_size,
    }
}

/// Returns the error code carried by a CPU MST failure.
fn mst_failure_code(error: &ChutoroError) -> Option<&str> {
    match error {
        ChutoroError::CpuMstFailure { code, .. } => Some(code.as_ref()),
        _ => None,
    }
}

/// Returns the subsystem message carried by a CPU MST failure.
fn mst_failure_message(error: &ChutoroError) -> Option<&str> {
    match error {
        ChutoroError::CpuMstFailure { message, .. } => Some(message.as_ref()),
        _ => None,
    }
}

/// Returns the error code carried by a CPU hierarchy failure.
fn hierarchy_failure_code(error: &ChutoroError) -> Option<&str> {
    match error {
        ChutoroError::CpuHierarchyFailure { code, .. } => Some(code.as_ref()),
        _ => None,
    }
}

#[test]
fn empty_session_publishes_an_empty_snapshot() {
    let outcome = rebuild_mst_labels("refresh-test", &inputs(0, &[], &[], MIN_CLUSTER_SIZE_TWO))
        .expect("an empty session must publish, not error");

    assert_eq!(outcome.labels, Vec::<usize>::new());
    assert_eq!(outcome.mst_backbone, Vec::<CandidateEdge>::new());
}

#[test]
fn non_contiguous_point_ids_report_the_highest_appended_index() {
    // Appending source indices 0, 1 and 3 leaves `core_distances` four long
    // while only three points exist, so point id 3 is not a valid node id.
    let core_distances = [1.0_f32, 1.0, 1.0, f32::INFINITY];
    let pending = [
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 3, 2.0, 1),
    ];

    let error = rebuild_mst_labels(
        "refresh-test",
        &inputs(3, &pending, &core_distances, MIN_CLUSTER_SIZE_TWO),
    )
    .expect_err("a non-contiguous session must be rejected");

    let expected_message = MstError::InvalidNodeId {
        node: 3,
        node_count: 3,
    }
    .to_string();
    // A structural matcher states the wrapper is a CPU MST failure, and the
    // code check pins the caller-input classification.
    assert_that!(error, matches_pattern!(ChutoroError::CpuMstFailure { .. }));
    assert_that!(mst_failure_code(&error), eq(Some("INVALID_NODE_ID")));
    assert_eq!(mst_failure_message(&error), Some(expected_message.as_str()));
}

#[test]
fn labels_cover_every_point_on_a_separated_fixture() {
    // Two tight pairs, far apart: the MST keeps both pairs apart, so the
    // hierarchy keeps two clusters and every point receives a label.
    let core_distances = [1.0_f32, 1.0, 1.0, 1.0];
    let pending = [
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(2, 3, 1.0, 1),
        CandidateEdge::new(1, 2, 10.0, 2),
    ];

    let outcome = rebuild_mst_labels(
        "refresh-test",
        &inputs(4, &pending, &core_distances, MIN_CLUSTER_SIZE_TWO),
    )
    .expect("the fixture must cluster");

    assert_eq!(outcome.labels.len(), 4);
    assert_ne!(outcome.labels.first(), outcome.labels.get(2));
    assert_eq!(
        outcome.mst_backbone.len(),
        3,
        "a connected four-node graph spans with three edges"
    );
}

/// OBL-7: duplicating one selected edge across a buffer pair changes nothing.
///
/// The three buffer pairs are enumerated rather than only the all-three case,
/// because a merge that double-counts would show up pairwise first. The
/// duplicated edge is the one Kruskal selects, so a merge that recovered it
/// twice would change the backbone rather than leaving it merely reordered.
#[rstest]
#[case::backbone_and_historical(BufferPair::BackboneHistorical)]
#[case::backbone_and_pending(BufferPair::BackbonePending)]
#[case::historical_and_pending(BufferPair::HistoricalPending)]
fn duplicated_edge_across_a_buffer_pair_leaves_the_result_unchanged(#[case] pair: BufferPair) {
    let core_distances = [1.0_f32, 1.0, 1.0];
    let baseline = [
        CandidateEdge::new(0, 1, 1.0, 3),
        CandidateEdge::new(1, 2, 2.0, 4),
    ];
    let expected = rebuild_mst_labels(
        "refresh-test",
        &inputs(3, &baseline, &core_distances, MIN_CLUSTER_SIZE_TWO),
    )
    .expect("the baseline fixture must cluster");
    assert!(
        !expected.mst_backbone.is_empty(),
        "test precondition: the duplicated edge must be MST-selected"
    );

    // One buffer holds the baseline; the pair's other member repeats it.
    let (mst_backbone, historical_edges, pending_edges): (
        &[CandidateEdge],
        &[CandidateEdge],
        &[CandidateEdge],
    ) = match pair {
        BufferPair::BackboneHistorical => (&baseline, &baseline, &[]),
        BufferPair::BackbonePending => (&baseline, &[], &baseline),
        BufferPair::HistoricalPending => (&[], &baseline, &baseline),
    };
    let duplicated = RefreshInputs {
        node_count: 3,
        mst_backbone,
        historical_edges,
        pending_edges,
        core_distances: &core_distances,
        min_cluster_size: MIN_CLUSTER_SIZE_TWO,
    };
    let actual = rebuild_mst_labels("refresh-test", &duplicated)
        .expect("the duplicated fixture must cluster");

    assert_eq!(actual.labels, expected.labels);
    assert_eq!(actual.mst_backbone, expected.mst_backbone);
}

/// The merge-buffer pair an overlap case duplicates across.
#[derive(Clone, Copy, Debug)]
enum BufferPair {
    /// `mst_backbone` and `historical_edges`.
    BackboneHistorical,
    /// `mst_backbone` and `pending_edges`.
    BackbonePending,
    /// `historical_edges` and `pending_edges`.
    HistoricalPending,
}

#[test]
fn historical_edges_participate_in_the_merge() {
    let core_distances = [1.0_f32, 1.0, 1.0];
    // On its own, `pending_edges` cannot connect node 0, so the forest would
    // have two components. The historical edge supplies the missing link.
    let historical = [
        CandidateEdge::new(0, 1, 1.0, 1),
        CandidateEdge::new(1, 2, 1.0, 2),
    ];
    let pending = [CandidateEdge::new(0, 1, 1.0, 3)];
    let retained = RefreshInputs {
        node_count: 3,
        mst_backbone: &[],
        historical_edges: &historical,
        pending_edges: &pending,
        core_distances: &core_distances,
        min_cluster_size: MIN_CLUSTER_SIZE_TWO,
    };

    let outcome =
        rebuild_mst_labels("refresh-test", &retained).expect("the merged fixture must cluster");

    assert_eq!(outcome.labels.len(), 3);
    assert_eq!(outcome.mst_backbone.len(), 2);
}

#[test]
fn non_finite_core_distances_are_rejected_by_kruskal() {
    let core_distances = [f32::INFINITY, 1.0];
    let pending = [CandidateEdge::new(0, 1, 1.0, 0)];

    let error = rebuild_mst_labels(
        "refresh-test",
        &inputs(2, &pending, &core_distances, MIN_CLUSTER_SIZE_TWO),
    )
    .expect_err("a non-finite weight must be rejected");

    assert_eq!(mst_failure_code(&error), Some("NON_FINITE_WEIGHT"));
}

#[test]
fn hierarchy_failure_is_mapped_not_panicked() {
    // Kruskal validates finiteness but not sign, so a negative weight reaches
    // hierarchy extraction, which rejects it. The rebuild must translate that
    // rejection rather than panic or silently publish.
    let core_distances = [-1.0_f32, -1.0];
    let pending = [CandidateEdge::new(0, 1, -5.0, 0)];

    let error = rebuild_mst_labels(
        "refresh-test",
        &inputs(2, &pending, &core_distances, MIN_CLUSTER_SIZE_TWO),
    )
    .expect_err("a negative MST weight must be rejected by hierarchy extraction");

    assert_eq!(hierarchy_failure_code(&error), Some("INVALID_EDGE_WEIGHT"));
}
