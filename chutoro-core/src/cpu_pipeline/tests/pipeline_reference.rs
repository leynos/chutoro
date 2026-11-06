//! End-to-end comparison of the production CPU pipeline with a sequential
//! reference.
//!
//! The fixture separates three groups of four points by five units while
//! spacing the members of each group a tenth of a unit apart. Any edge the
//! parallel transform or the slice-based Kruskal path drops, and any
//! mutual-reachability weight it corrupts, therefore changes the forest the
//! reference accepts instead of being absorbed by ties.

use std::collections::BTreeMap;

use crate::mst::property::oracle::sequential_kruskal_edges;
use crate::{ClusterId, DataSource, HnswParams};

use super::super::{
    build_index_and_core_distances, build_mutual_reachability_forest, run_cpu_pipeline_with_len,
};
use super::fixture::{ClusteredSource, EXPECTED_CLUSTERS, ITEM_COUNT, MIN_CLUSTER_SIZE};
use super::reference::sequential_mutual_edges;

#[test]
fn production_forest_matches_the_sequential_reference() {
    let source = ClusteredSource::new();
    let params = HnswParams::new(4, 16).expect("fixture parameters must be valid");

    let (harvested, core_distances) =
        build_index_and_core_distances(&source, ITEM_COUNT, MIN_CLUSTER_SIZE, &params)
            .expect("fixture index build must succeed");

    let production =
        build_mutual_reachability_forest(source.name(), ITEM_COUNT, &harvested, &core_distances)
            .expect("production forest must be built");

    let reference_edges = sequential_mutual_edges(&harvested, &core_distances)
        .expect("every fixture endpoint has a core distance");
    let reference = sequential_kruskal_edges(ITEM_COUNT, &reference_edges);

    let production_edges: Vec<(usize, usize, f32, u64)> = production
        .edges()
        .iter()
        .map(|edge| (edge.source(), edge.target(), edge.weight(), edge.sequence()))
        .collect();

    assert_eq!(
        production_edges, reference,
        "the production forest must match the sequential reference edge for edge"
    );
}

#[test]
fn production_pipeline_recovers_the_fixture_clusters() {
    let source = ClusteredSource::new();
    let params = HnswParams::new(4, 16).expect("fixture parameters must be valid");

    let result = run_cpu_pipeline_with_len(&source, ITEM_COUNT, MIN_CLUSTER_SIZE, &params)
        .expect("production pipeline must succeed");

    assert_eq!(
        result.cluster_count(),
        EXPECTED_CLUSTERS,
        "the separated fixture must yield one cluster per group"
    );
    assert_eq!(
        canonical_partition(result.assignments()),
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]],
        "each group of four adjacent points must form one cluster"
    );
}

/// Groups item indices by cluster identifier, ignoring label numbering.
fn canonical_partition(assignments: &[ClusterId]) -> Vec<Vec<usize>> {
    let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, assignment) in assignments.iter().enumerate() {
        groups.entry(assignment.get()).or_default().push(index);
    }

    let mut partition: Vec<Vec<usize>> = groups.into_values().collect();
    partition.sort_unstable();
    partition
}
