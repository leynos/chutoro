//! Unit tests for hierarchy extraction from the mutual-reachability MST.

use std::num::NonZeroUsize;

use rstest::rstest;

use crate::{
    CandidateEdge, EdgeHarvest, HierarchyConfig, HierarchyError, HierarchyErrorCode,
    extract_labels_from_mst, parallel_kruskal,
};

fn core_distances_1d(points: &[f32], min_cluster_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(points.len());
    for (idx, &value) in points.iter().enumerate() {
        let mut distances: Vec<f32> = points
            .iter()
            .enumerate()
            .filter_map(|(j, &other)| {
                (j != idx).then_some(value.mul_add(1.0, std::ops::Neg::neg(other)).abs())
            })
            .collect();
        distances.sort_by(f32::total_cmp);
        let core = distances
            .get(min_cluster_size.saturating_sub(1))
            .copied()
            .or_else(|| distances.last().copied())
            .unwrap_or(0.0);
        out.push(core);
    }
    out
}

fn mutual_reachability_edges_1d(points: &[f32], min_cluster_size: usize) -> EdgeHarvest {
    let core = core_distances_1d(points, min_cluster_size);
    let mut edges = Vec::new();
    let mut seq = 0u64;
    for (i, (&left, &left_core_distance)) in points.iter().zip(core.iter()).enumerate() {
        for (j, (&right, &right_core_distance)) in points
            .iter()
            .zip(core.iter())
            .enumerate()
            .skip(i.saturating_add(1))
        {
            let dist = left.mul_add(1.0, std::ops::Neg::neg(right)).abs();
            let weight = dist.max(left_core_distance).max(right_core_distance);
            edges.push(CandidateEdge::new(i, j, weight, seq));
            seq += 1;
        }
    }
    EdgeHarvest::new(edges)
}

fn unique_label_count(labels: &[usize]) -> usize {
    use std::collections::HashSet;

    labels.iter().copied().collect::<HashSet<_>>().len()
}

#[rstest]
#[case(vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2], 2, 2)]
#[case(vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0], 2, 2)]
fn extracts_two_clusters_without_noise(
    #[case] points: Vec<f32>,
    #[case] min_cluster_size: usize,
    #[case] expected_clusters: usize,
) {
    let harvest = mutual_reachability_edges_1d(&points, min_cluster_size);
    let forest = parallel_kruskal(points.len(), &harvest).expect("MST should succeed");

    let labels = extract_labels_from_mst(
        points.len(),
        forest.edges(),
        HierarchyConfig::new(NonZeroUsize::new(min_cluster_size).expect("non-zero")),
    )
    .expect("hierarchy extraction should succeed");

    assert_eq!(unique_label_count(&labels), expected_clusters);
    let [first, second, third, fourth, fifth, sixth] = labels.as_slice() else {
        panic!("expected six labels for two three-point clusters");
    };
    assert_eq!(first, second);
    assert_eq!(second, third);
    assert_ne!(third, fourth);
    assert_eq!(fourth, fifth);
    assert_eq!(fifth, sixth);
}

#[test]
fn assigns_outlier_to_noise_when_min_cluster_size_excludes_it() {
    let points = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2, 100.0];
    let min_cluster_size = 2;
    let harvest = mutual_reachability_edges_1d(&points, min_cluster_size);
    let forest = parallel_kruskal(points.len(), &harvest).expect("MST should succeed");

    let labels = extract_labels_from_mst(
        points.len(),
        forest.edges(),
        HierarchyConfig::new(NonZeroUsize::new(min_cluster_size).expect("non-zero")),
    )
    .expect("hierarchy extraction should succeed");

    let clusters = unique_label_count(&labels);
    assert_eq!(clusters, 3, "expected two clusters plus noise");

    let noise_label = *labels.iter().max().expect("non-empty labels");
    let outlier_label = labels.last().copied().expect("outlier label must exist");
    assert_eq!(
        outlier_label, noise_label,
        "outlier should be classified as noise"
    );
}

#[test]
fn assigns_all_points_to_noise_when_every_component_is_too_small() {
    let node_count = 4;
    let min_cluster_size = 3;

    let labels = extract_labels_from_mst(
        node_count,
        &[],
        HierarchyConfig::new(NonZeroUsize::new(min_cluster_size).expect("non-zero")),
    )
    .expect("hierarchy extraction should succeed for empty forests");

    assert_eq!(labels, vec![0; node_count]);
}

#[test]
fn rejects_empty_dataset() {
    let err = extract_labels_from_mst(
        0,
        &[],
        HierarchyConfig::new(NonZeroUsize::new(2).expect("non-zero")),
    )
    .expect_err("empty datasets are invalid");
    assert!(matches!(err, HierarchyError::EmptyDataset));
}

#[test]
fn rejects_min_cluster_size_larger_than_dataset() {
    let err = extract_labels_from_mst(
        2,
        &[],
        HierarchyConfig::new(NonZeroUsize::new(3).expect("non-zero")),
    )
    .expect_err("min_cluster_size > node_count is invalid");
    assert!(matches!(err, HierarchyError::MinClusterSizeTooLarge { .. }));
}

#[test]
fn rejects_negative_edge_weights() {
    let edges = EdgeHarvest::new(vec![CandidateEdge::new(0, 1, -1.0, 0)]);
    let forest = parallel_kruskal(2, &edges).expect("MST should accept finite weights");

    let err = extract_labels_from_mst(
        2,
        forest.edges(),
        HierarchyConfig::new(NonZeroUsize::new(2).expect("non-zero")),
    )
    .expect_err("negative edge weights are invalid for hierarchy extraction");

    assert!(matches!(err, HierarchyError::InvalidEdgeWeight { .. }));
}

#[test]
fn rejects_mst_endpoint_outside_hierarchy_dataset() {
    let edges = EdgeHarvest::new(vec![CandidateEdge::new(0, 2, 1.0, 0)]);
    let forest = parallel_kruskal(3, &edges).expect("MST should accept its own valid endpoint");

    let err = extract_labels_from_mst(
        2,
        forest.edges(),
        HierarchyConfig::new(NonZeroUsize::new(2).expect("non-zero")),
    )
    .expect_err("hierarchy endpoint must belong to the hierarchy dataset");

    assert_eq!(
        err,
        HierarchyError::InvalidEdgeEndpoint {
            endpoint: 2,
            node_count: 2,
        }
    );
    assert_eq!(err.code(), HierarchyErrorCode::InvalidEdgeEndpoint);
    assert_eq!(err.code().as_str(), "INVALID_EDGE_ENDPOINT");
}

#[rstest]
#[case(
    HierarchyError::EmptyDataset,
    HierarchyErrorCode::EmptyDataset,
    "EMPTY_DATASET"
)]
#[case(
    HierarchyError::MinClusterSizeTooLarge {
        node_count: 2,
        min_cluster_size: 3,
    },
    HierarchyErrorCode::MinClusterSizeTooLarge,
    "MIN_CLUSTER_SIZE_TOO_LARGE"
)]
#[case(
    HierarchyError::InvalidEdgeWeight {
        left: 0,
        right: 1,
        weight: -1.0,
    },
    HierarchyErrorCode::InvalidEdgeWeight,
    "INVALID_EDGE_WEIGHT"
)]
#[case(
    HierarchyError::InvalidEdgeEndpoint {
        endpoint: 2,
        node_count: 2,
    },
    HierarchyErrorCode::InvalidEdgeEndpoint,
    "INVALID_EDGE_ENDPOINT"
)]
#[case(
    HierarchyError::InvalidForestReference { node_id: 3 },
    HierarchyErrorCode::InvalidForestReference,
    "INVALID_FOREST_REFERENCE"
)]
#[case(
    HierarchyError::InvalidClusterReference { cluster_id: 4 },
    HierarchyErrorCode::InvalidClusterReference,
    "INVALID_CLUSTER_REFERENCE"
)]
#[case(
    HierarchyError::InvalidPointReference {
        point_id: 5,
        node_count: 2,
    },
    HierarchyErrorCode::InvalidPointReference,
    "INVALID_POINT_REFERENCE"
)]
fn hierarchy_error_codes_are_stable(
    #[case] error: HierarchyError,
    #[case] expected_code: HierarchyErrorCode,
    #[case] expected_name: &'static str,
) {
    let code = error.code();

    assert_eq!(code, expected_code);
    assert_eq!(code.as_str(), expected_name);
}
