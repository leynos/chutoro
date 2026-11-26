//! Backbone selection and recovery in the pure rebuild.
//!
//! These cases pin which raw candidates the MST selects under a given set of
//! core distances, and how recovery maps the accepted forest edges back to
//! their raw inputs.

use pretty_assertions::assert_eq;

use super::{RefreshInputs, inputs, min_cluster_size, rebuild_mst_labels};
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

    // Recovery returns the backbone sorted by canonical key, so `(0, 4)`
    // precedes `(1, 2)` here even though it appears later in the input.
    assert_eq!(
        raised.mst_backbone,
        vec![edges[0], edges[4], edges[1], edges[3]],
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
    // The input edges are written with reversed orientation on purpose; the
    // backbone stores them canonically, because orientation is not part of an
    // undirected edge's meaning but *is* part of its derived equality.
    assert_eq!(
        outcome.mst_backbone.first(),
        Some(&CandidateEdge::new(0, 1, 1.0, 7)),
        "the retained edge keeps its raw distance and canonical orientation"
    );
    assert_eq!(
        outcome.mst_backbone.get(1),
        Some(&CandidateEdge::new(1, 2, 2.0, 8))
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

/// Tied reweighted weights must not let buffer order choose the raw distance.
///
/// With core distance 10 on every node, both `(0, 1, 9, 7)` and
/// `(0, 1, 1, 7)` reweight to 10, so the accepted weight alone cannot tell them
/// apart. Recovery must pick deterministically — it keeps the smallest raw
/// distance — and must produce the same backbone whichever buffer supplied a
/// candidate. A first-match rule would return the distance-9 edge when it
/// appears first and the distance-1 edge when the distance-1 candidate is
/// duplicated into an earlier buffer, which is exactly the OBL-7 violation
/// this case rules out.
#[test]
fn tied_reweighted_candidates_recover_deterministically() {
    let core_distances = [10.0_f32, 10.0, 10.0];
    let heavy = CandidateEdge::new(0, 1, 9.0, 7);
    let light = CandidateEdge::new(0, 1, 1.0, 7);
    let other = CandidateEdge::new(1, 2, 2.0, 8);

    // Baseline: the heavy candidate comes first.
    let heavy_first = [heavy, light, other];
    let baseline = rebuild_mst_labels("refresh-test", &inputs(3, &heavy_first, &core_distances, 2))
        .expect("the tied fixture must cluster");

    // Same set, but the light candidate now also reaches the merge through the
    // earlier backbone buffer.
    let duplicated = RefreshInputs {
        node_count: 3,
        mst_backbone: &[light],
        historical_edges: &[],
        pending_edges: &heavy_first,
        core_distances: &core_distances,
        min_cluster_size: min_cluster_size(2),
    };
    let actual = rebuild_mst_labels("refresh-test", &duplicated)
        .expect("the duplicated tied fixture must cluster");

    assert_eq!(
        actual.mst_backbone, baseline.mst_backbone,
        "buffer overlap must not change the recovered backbone"
    );
    let retained = actual
        .mst_backbone
        .iter()
        .find(|edge| edge.sequence() == 7)
        .expect("the shared-key edge must be retained");
    assert!(
        retained.distance().total_cmp(&1.0_f32).is_eq(),
        "recovery must keep the smallest raw distance among tied candidates, got {retained:?}"
    );
}
