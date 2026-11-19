//! Error-boundary tests for the CPU pipeline's MST and hierarchy stages.
//!
//! Pins the translation from `MstError` to `ChutoroError::CpuMstFailure` and
//! from `HierarchyError` to `ChutoroError::CpuHierarchyFailure`, and exercises
//! the real pipeline boundary in explicit one-thread and eight-thread pools, so
//! a nondeterministic error selection would surface here rather than only in a
//! direct MST test.

use rayon::ThreadPoolBuilder;
use rstest::rstest;

use crate::{CandidateEdge, ChutoroError, ChutoroErrorCode, EdgeHarvest, HierarchyError, MstError};

use super::super::{build_mutual_reachability_forest, map_cpu_hierarchy_error, map_cpu_mst_error};

/// Thread counts the pipeline boundary is pinned in.
const THREAD_COUNTS: [usize; 2] = [1, 8];

/// Extracts a `CpuMstFailure`'s inner code and message.
///
/// Returns an error rather than panicking so each test keeps its own
/// assertions at the `#[test]` boundary.
fn cpu_mst_failure_fields(error: &ChutoroError) -> Result<(&str, &str), String> {
    if error.code() != ChutoroErrorCode::CpuMstFailure {
        return Err(format!("expected CpuMstFailure, got {error:?}"));
    }
    match error {
        ChutoroError::CpuMstFailure { code, message } => Ok((code, message)),
        other => Err(format!("expected CpuMstFailure, got {other:?}")),
    }
}

/// Extracts a `CpuHierarchyFailure`'s inner code and message.
///
/// Returns an error rather than panicking so each test keeps its own
/// assertions at the `#[test]` boundary.
fn cpu_hierarchy_failure_fields(error: &ChutoroError) -> Result<(&str, &str), String> {
    if error.code() != ChutoroErrorCode::CpuHierarchyFailure {
        return Err(format!("expected CpuHierarchyFailure, got {error:?}"));
    }
    match error {
        ChutoroError::CpuHierarchyFailure { code, message } => Ok((code, message)),
        other => Err(format!("expected CpuHierarchyFailure, got {other:?}")),
    }
}

#[rstest]
#[case(MstError::EmptyGraph, "EMPTY_GRAPH")]
#[case(
    MstError::InvalidNodeId {
        node: 5,
        node_count: 3
    },
    "INVALID_NODE_ID"
)]
#[case(
    MstError::NonFiniteWeight { left: 1, right: 2 },
    "NON_FINITE_WEIGHT"
)]
#[case(
    MstError::LockPoisoned {
        resource: "union-find root lock"
    },
    "LOCK_POISONED"
)]
#[case(
    MstError::InvariantViolation {
        invariant: "root index in range",
        index: 4,
        lock_count: 3
    },
    "INVARIANT_VIOLATION"
)]
fn map_cpu_mst_error_forwards_every_variant(#[case] error: MstError, #[case] expected_code: &str) {
    let mapped = map_cpu_mst_error(&error);
    let (code, message) =
        cpu_mst_failure_fields(&mapped).expect("mapper must produce a CpuMstFailure");

    assert_eq!(mapped.code(), ChutoroErrorCode::CpuMstFailure);
    assert_eq!(code, expected_code);
    assert_eq!(code, error.code().as_str());
    assert_eq!(message, error.to_string());
}

#[rstest]
#[case(HierarchyError::EmptyDataset, "EMPTY_DATASET")]
#[case(
    HierarchyError::MinClusterSizeTooLarge {
        node_count: 3,
        min_cluster_size: 4
    },
    "MIN_CLUSTER_SIZE_TOO_LARGE"
)]
#[case(
    HierarchyError::InvalidEdgeWeight {
        left: 0,
        right: 1,
        weight: f32::NAN
    },
    "INVALID_EDGE_WEIGHT"
)]
#[case(
    HierarchyError::InvalidEdgeEndpoint {
        endpoint: 7,
        node_count: 3
    },
    "INVALID_EDGE_ENDPOINT"
)]
#[case(
    HierarchyError::InvalidForestReference { node_id: 9 },
    "INVALID_FOREST_REFERENCE"
)]
#[case(
    HierarchyError::InvalidClusterReference { cluster_id: 2 },
    "INVALID_CLUSTER_REFERENCE"
)]
#[case(
    HierarchyError::InvalidPointReference {
        point_id: 5,
        node_count: 3
    },
    "INVALID_POINT_REFERENCE"
)]
fn map_cpu_hierarchy_error_forwards_every_variant(
    #[case] error: HierarchyError,
    #[case] expected_code: &str,
) {
    let mapped = map_cpu_hierarchy_error(&error);
    let (code, message) =
        cpu_hierarchy_failure_fields(&mapped).expect("mapper must produce a CpuHierarchyFailure");

    assert_eq!(mapped.code(), ChutoroErrorCode::CpuHierarchyFailure);
    assert_eq!(code, expected_code);
    assert_eq!(code, error.code().as_str());
    assert_eq!(message, error.to_string());
}

#[test]
fn build_mutual_reachability_forest_reports_the_first_weighted_edge() {
    // `items` is the Kruskal node count, while the core-distance vector is
    // longer so the weighting transform accepts endpoints that the MST stage
    // must then reject. The first harvested edge is the one at sequence 0.
    let items = 2;
    let core_distances = [0.0_f32; 5];
    let harvest = EdgeHarvest::new(vec![
        CandidateEdge::new(0, 3, 1.0, 0),
        CandidateEdge::new(4, 1, 1.0, 1),
    ]);

    for threads in THREAD_COUNTS {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("explicit Rayon pool must build");
        let error = pool
            .install(|| {
                build_mutual_reachability_forest("fixture", items, &harvest, &core_distances)
            })
            .expect_err("out-of-range harvested endpoints must fail");

        let (code, message) =
            cpu_mst_failure_fields(&error).expect("pipeline must produce a CpuMstFailure");

        assert_eq!(code, "INVALID_NODE_ID", "pipeline with {threads} thread(s)");
        assert_eq!(
            message, "edge references node 3, but node_count is 2",
            "pipeline with {threads} thread(s)"
        );
    }
}
