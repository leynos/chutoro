//! Unit tests for the pure refresh rebuild.
//!
//! Every case here drives [`super::rebuild_mst_labels`] directly: no HNSW
//! index, no session, no clock. The fixtures are hand-built raw edge lists so
//! the expected labels and backbone are derivable by reading the case.

use std::num::NonZeroUsize;

use googletest::prelude::*;
use pretty_assertions::assert_eq;
use rstest::rstest;

use super::{RefreshInputs, rebuild_mst_labels};
use crate::{CandidateEdge, ChutoroError, MstError};

/// Builds the standard non-zero `min_cluster_size` used by these cases.
fn min_cluster_size(size: usize) -> NonZeroUsize {
    NonZeroUsize::new(size).expect("test min_cluster_size must be non-zero")
}

/// Builds inputs with no retained edges, so a case states only its pending set.
fn inputs<'a>(
    node_count: usize,
    pending_edges: &'a [CandidateEdge],
    core_distances: &'a [f32],
    min_size: usize,
) -> RefreshInputs<'a> {
    RefreshInputs {
        node_count,
        mst_backbone: &[],
        historical_edges: &[],
        pending_edges,
        core_distances,
        min_cluster_size: min_cluster_size(min_size),
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
    let outcome = rebuild_mst_labels("refresh-test", &inputs(0, &[], &[], 2))
        .expect("an empty session must publish, not error");

    assert_eq!(outcome.labels, Vec::<usize>::new());
    assert_eq!(outcome.mst_backbone, Vec::<CandidateEdge>::new());
}

/// Warm-up publishes all-noise labels *and* keeps its candidate pool.
///
/// The two claims belong to one case because they are the same decision: no
/// tree exists below `min_cluster_size`, so the labels carry no structure and
/// the harvested candidates must ride forward instead. HNSW harvests an edge
/// only when the later endpoint is inserted, so an old-old candidate is never
/// re-harvested; dropping the pool would lose those connections for the whole
/// session and the first real rebuild could not see them.
#[test]
fn below_min_cluster_size_publishes_all_noise_and_retains_candidates() {
    let core_distances = [1.0_f32, 1.0, 1.0, 1.0];
    let pending = [CandidateEdge::new(0, 1, 1.0, 0)];

    let outcome = rebuild_mst_labels("refresh-test", &inputs(4, &pending, &core_distances, 5))
        .expect("a warm-up session must publish all-noise labels");

    assert_eq!(outcome.labels, vec![0, 0, 0, 0]);
    assert_eq!(outcome.mst_backbone, pending.to_vec());
}

/// The current core distances must change which edges the MST selects.
///
/// A five-node path 0-1-2-3-4 of unit edges plus a 1.5 shortcut 0-4 is the
/// fixture. With flat core distances the path wins and the shortcut is
/// rejected. Raising node 2's core distance reweights both edges incident on
/// it to 10, so Kruskal takes the shortcut instead and the backbone changes.
/// Skipping the reweight, or feeding it stale core distances, leaves the flat
/// backbone in place and fails both assertions.
#[test]
fn core_distances_change_the_selected_backbone() {
    let edges = [
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 1.0, 1),
        CandidateEdge::new(2, 3, 1.0, 2),
        CandidateEdge::new(3, 4, 1.0, 3),
        CandidateEdge::new(0, 4, 1.5, 4),
    ];

    let flat_cores = [0.0_f32; 5];
    let flat = rebuild_mst_labels("refresh-test", &inputs(5, &edges, &flat_cores, 2))
        .expect("the flat-core fixture must cluster");

    assert_eq!(
        flat.mst_backbone,
        vec![edges[0], edges[1], edges[2], edges[3]],
        "flat cores must select the unit path and reject the shortcut"
    );

    // Node 2 sits on the path but not on the shortcut.
    let raised_cores = [0.0_f32, 0.0, 10.0, 0.0, 0.0];
    let raised = rebuild_mst_labels("refresh-test", &inputs(5, &edges, &raised_cores, 2))
        .expect("the raised-core fixture must cluster");

    assert_eq!(
        raised.mst_backbone,
        vec![edges[0], edges[1], edges[3], edges[4]],
        "raising node 2's core distance must reweight its edges out of the MST"
    );
    assert_eq!(flat.labels.len(), 5);
    assert_eq!(raised.labels.len(), 5);
}

/// Crossed warm-up: the retained pool feeds the first real rebuild.
#[test]
fn crossing_min_cluster_size_keeps_the_warm_up_candidates() {
    // The pool must only reference nodes that exist at warm-up time: a
    // four-point session has ids 0..4, so the pool spans 0-1 and 1-2.
    let warm_up_cores = [1.0_f32, 1.0, 1.0, 1.0];
    let warm_up_pool = [
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 1.0, 1),
    ];

    // Warm-up: four points, one below the threshold of five.
    let warmed = rebuild_mst_labels("refresh-test", &inputs(4, &warm_up_pool, &warm_up_cores, 5))
        .expect("warm-up must publish");
    assert_eq!(warmed.labels, vec![0; 4]);
    assert_eq!(warmed.mst_backbone, warm_up_pool.to_vec());

    // The fifth point crosses the threshold. The retained pool enters the merge
    // as the backbone, so both warm-up edges reach the rebuilt forest instead of
    // being limited to the single newly harvested edge.
    let crossed_cores = [1.0_f32, 1.0, 1.0, 1.0, 1.0];
    let crossed = RefreshInputs {
        node_count: 5,
        mst_backbone: &warmed.mst_backbone,
        historical_edges: &[],
        pending_edges: &[CandidateEdge::new(2, 3, 1.0, 2)],
        core_distances: &crossed_cores,
        min_cluster_size: min_cluster_size(5),
    };
    let rebuilt =
        rebuild_mst_labels("refresh-test", &crossed).expect("the first real rebuild must succeed");

    assert_eq!(rebuilt.labels.len(), 5);
    let retained = rebuilt
        .mst_backbone
        .iter()
        .filter(|edge| warm_up_pool.contains(edge))
        .count();
    assert_eq!(
        retained, 2,
        "both warm-up candidates must reach the rebuilt forest, got {:?}",
        rebuilt.mst_backbone
    );
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

    let error = rebuild_mst_labels("refresh-test", &inputs(3, &pending, &core_distances, 2))
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

    let outcome = rebuild_mst_labels("refresh-test", &inputs(4, &pending, &core_distances, 2))
        .expect("the fixture must cluster");

    assert_eq!(outcome.labels.len(), 4);
    assert_ne!(outcome.labels.first(), outcome.labels.get(2));
    assert_eq!(
        outcome.mst_backbone.len(),
        3,
        "a connected four-node graph spans with three edges"
    );
}

#[test]
fn backbone_recovery_matches_the_forest_for_canonical_inputs() {
    let core_distances = [1.0_f32, 1.0, 1.0];
    // The (1, 0) edge is deliberately reversed: the backbone must come back
    // as the raw input, not as Kruskal's canonicalized form.
    let pending = [
        CandidateEdge::new(1, 0, 1.0, 7),
        CandidateEdge::new(2, 1, 2.0, 8),
        CandidateEdge::new(0, 2, 9.0, 9),
    ];

    let outcome = rebuild_mst_labels("refresh-test", &inputs(3, &pending, &core_distances, 2))
        .expect("the fixture must cluster");

    assert_eq!(outcome.mst_backbone.len(), 2);
    assert_eq!(
        outcome.mst_backbone.first(),
        Some(&CandidateEdge::new(1, 0, 1.0, 7)),
        "the retained edge keeps its raw orientation and raw distance"
    );
    assert_eq!(
        outcome.mst_backbone.get(1),
        Some(&CandidateEdge::new(2, 1, 2.0, 8))
    );
    assert!(
        !outcome.mst_backbone.iter().any(|edge| edge.sequence() == 9),
        "the heaviest edge is not in the MST"
    );
}

#[test]
fn backbone_retains_raw_distances_not_mutual_reachability_weights() {
    // The raw distance is 1.0 everywhere, but core distances of 4.0 raise
    // every weight. A retained edge must still carry its raw distance so the
    // next refresh cannot inherit this refresh's inflated weight.
    let core_distances = [4.0_f32, 4.0, 4.0];
    let pending = [
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 1.0, 1),
        CandidateEdge::new(0, 2, 1.0, 2),
    ];

    let outcome = rebuild_mst_labels("refresh-test", &inputs(3, &pending, &core_distances, 2))
        .expect("the fixture must cluster");

    for edge in &outcome.mst_backbone {
        assert!(
            edge.distance() <= 1.0,
            "backbone edge {edge:?} kept a mutual-reachability weight"
        );
    }
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
    let expected = rebuild_mst_labels("refresh-test", &inputs(3, &baseline, &core_distances, 2))
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
        min_cluster_size: min_cluster_size(2),
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
        min_cluster_size: min_cluster_size(2),
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

    let error = rebuild_mst_labels("refresh-test", &inputs(2, &pending, &core_distances, 2))
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

    let error = rebuild_mst_labels("refresh-test", &inputs(2, &pending, &core_distances, 2))
        .expect_err("a negative MST weight must be rejected by hierarchy extraction");

    assert_eq!(hierarchy_failure_code(&error), Some("INVALID_EDGE_WEIGHT"));
}
