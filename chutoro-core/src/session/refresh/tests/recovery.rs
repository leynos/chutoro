//! Backbone selection and recovery in the pure rebuild.
//!
//! These cases pin which raw candidates the MST selects under a given set of
//! core distances, and how recovery maps the accepted forest edges back to
//! their raw inputs.

use pretty_assertions::assert_eq;

use super::{inputs, rebuild_mst_labels};
use crate::CandidateEdge;

/// Two same-key candidates with different raw distances must not be confused.
///
/// Kruskal picks the lighter reweighted edge; recovery must return *that*
/// edge's raw input, not the first candidate sharing its endpoint and sequence
/// key. Here `(0, 1, 1.0, 7)` and `(0, 1, 9.0, 7)` share a key, so a key-only
/// recovery could retain the raw distance 9 while the forest carries weight 1.
#[test]
fn recovery_matches_by_weight_within_a_shared_key() {
    let core_distances = [0.0_f32, 0.0, 0.0];
    let candidates = [
        CandidateEdge::new(0, 1, 9.0, 7),
        CandidateEdge::new(0, 1, 1.0, 7),
        CandidateEdge::new(1, 2, 2.0, 8),
    ];

    let outcome = rebuild_mst_labels("refresh-test", &inputs(3, &candidates, &core_distances, 2))
        .expect("the shared-key fixture must cluster");

    assert!(
        outcome
            .mst_backbone
            .iter()
            .any(|edge| edge.sequence() == 7 && edge.distance().total_cmp(&1.0_f32).is_eq()),
        "recovery must retain the selected edge's raw distance, got {:?}",
        outcome.mst_backbone
    );
    assert!(
        !outcome
            .mst_backbone
            .iter()
            .any(|edge| edge.sequence() == 7 && edge.distance().total_cmp(&9.0_f32).is_eq()),
        "recovery must not retain the unselected same-key candidate"
    );
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

    // Without this the loop below would pass vacuously on an empty backbone.
    assert_eq!(
        outcome.mst_backbone.len(),
        2,
        "a three-node connected fixture must retain two backbone edges"
    );
    for edge in &outcome.mst_backbone {
        assert!(
            edge.distance().total_cmp(&1.0_f32).is_eq(),
            "backbone edge {edge:?} kept a mutual-reachability weight"
        );
    }
}
