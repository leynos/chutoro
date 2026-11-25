//! Warm-up and crossing behaviour of the pure rebuild.
//!
//! Warm-up is the phase below `min_cluster_size`, where no tree is selected but
//! the harvested candidates must survive to the first real rebuild. These
//! cases pin that contract and the overlap tolerance it inherits from OBL-7.

use pretty_assertions::assert_eq;

use super::{RefreshInputs, inputs, min_cluster_size, rebuild_mst_labels};
use crate::CandidateEdge;

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

/// OBL-7 applies to the warm-up pool as well as to the real rebuild.
///
/// Warm-up returns the merged pool instead of running Kruskal, so it must
/// apply the same overlap tolerance explicitly. The same raw edge reaching the
/// merge through all three buffers must be retained once, not three times;
/// otherwise the pool grows with every warm-up refresh.
#[test]
fn warm_up_retains_an_overlapping_candidate_once() {
    let core_distances = [1.0_f32, 1.0, 1.0, 1.0];
    let shared = [CandidateEdge::new(0, 1, 1.0, 0)];
    let expected = rebuild_mst_labels("refresh-test", &inputs(4, &shared, &core_distances, 5))
        .expect("the baseline warm-up must publish");

    let overlapping = RefreshInputs {
        node_count: 4,
        mst_backbone: &shared,
        historical_edges: &shared,
        pending_edges: &shared,
        core_distances: &core_distances,
        min_cluster_size: min_cluster_size(5),
    };
    let actual = rebuild_mst_labels("refresh-test", &overlapping)
        .expect("the overlapping warm-up must publish");

    assert_eq!(actual.labels, expected.labels);
    assert_eq!(
        actual.mst_backbone, expected.mst_backbone,
        "an overlapping warm-up must not duplicate a candidate"
    );
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
