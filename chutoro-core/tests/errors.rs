//! Integration tests covering the public error types exposed by `chutoro-core`.

use std::{num::NonZeroUsize, sync::Arc};

use chutoro_core::{
    ChutoroError, ChutoroErrorCode, DataSourceError, DataSourceErrorCode, ExecutionStrategy,
};
#[cfg(feature = "cpu")]
use chutoro_core::{
    HierarchyError, HierarchyErrorCode, HnswError, HnswErrorCode, MstError, MstErrorCode,
};
use rstest::rstest;

/// Assert the contract every generated code enum shares: the error maps to the
/// expected code, `as_str` returns the stable name, and `Display` agrees.
#[cfg(feature = "cpu")]
macro_rules! assert_code_contract {
    ($error:expr, $expected_code:expr, $expected_name:expr) => {{
        let error = $error;
        let expected_code = $expected_code;
        assert_eq!(error.code(), expected_code);
        assert_eq!(expected_code.as_str(), $expected_name);
        assert_eq!(expected_code.to_string(), $expected_name);
    }};
}

#[rstest]
#[case(DataSourceError::OutOfBounds { index: 0 }, DataSourceErrorCode::OutOfBounds)]
#[case(
    DataSourceError::OutputLengthMismatch { out: 1, expected: 2 },
    DataSourceErrorCode::OutputLengthMismatch,
)]
#[case(
    DataSourceError::DimensionMismatch { left: 1, right: 2 },
    DataSourceErrorCode::DimensionMismatch,
)]
#[case(DataSourceError::EmptyData, DataSourceErrorCode::EmptyData)]
#[case(DataSourceError::ZeroDimension, DataSourceErrorCode::ZeroDimension)]
fn returns_expected_data_source_code(
    #[case] error: DataSourceError,
    #[case] expected: DataSourceErrorCode,
) {
    assert_eq!(error.code(), expected);
    assert_eq!(error.code().as_str(), expected.as_str());
    assert_eq!(expected.to_string(), expected.as_str());
}

#[rstest]
#[case(
    ChutoroError::InvalidMinClusterSize { got: 0 },
    ChutoroErrorCode::InvalidMinClusterSize,
    None,
)]
#[case(
    ChutoroError::EmptySource { data_source: Arc::from("empty") },
    ChutoroErrorCode::EmptySource,
    None,
)]
#[case(
    ChutoroError::InsufficientItems {
        data_source: Arc::from("small"),
        items: 3,
        min_cluster_size: NonZeroUsize::new(5).expect("non-zero"),
    },
    ChutoroErrorCode::InsufficientItems,
    None,
)]
#[case(
    ChutoroError::BackendUnavailable {
        requested: ExecutionStrategy::Auto,
    },
    ChutoroErrorCode::BackendUnavailable,
    None,
)]
#[case(
    ChutoroError::DataSource {
        data_source: Arc::from("source"),
        error: DataSourceError::OutOfBounds { index: 1 },
    },
    ChutoroErrorCode::DataSourceFailure,
    Some(DataSourceErrorCode::OutOfBounds),
)]
fn returns_expected_chutoro_code(
    #[case] error: ChutoroError,
    #[case] expected: ChutoroErrorCode,
    #[case] data_source_code: Option<DataSourceErrorCode>,
) {
    assert_eq!(error.code(), expected);
    assert_eq!(error.code().as_str(), expected.as_str());
    assert_eq!(expected.to_string(), expected.as_str());
    assert_eq!(error.data_source_code(), data_source_code);
}

#[cfg(feature = "cpu")]
#[rstest]
#[case(MstError::EmptyGraph, MstErrorCode::EmptyGraph, "EMPTY_GRAPH")]
#[case(
    MstError::InvalidNodeId {
        node: 1,
        node_count: 2,
    },
    MstErrorCode::InvalidNodeId,
    "INVALID_NODE_ID"
)]
#[case(
    MstError::NonFiniteWeight { left: 0, right: 1 },
    MstErrorCode::NonFiniteWeight,
    "NON_FINITE_WEIGHT"
)]
#[case(
    MstError::LockPoisoned { resource: "graph" },
    MstErrorCode::LockPoisoned,
    "LOCK_POISONED"
)]
#[case(
    MstError::InvariantViolation {
        invariant: "order",
        index: 0,
        lock_count: 1,
    },
    MstErrorCode::InvariantViolation,
    "INVARIANT_VIOLATION"
)]
fn returns_expected_mst_code(
    #[case] error: MstError,
    #[case] expected: MstErrorCode,
    #[case] expected_name: &'static str,
) {
    assert_code_contract!(error, expected, expected_name);
}

#[cfg(feature = "cpu")]
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
fn returns_expected_hierarchy_code(
    #[case] error: HierarchyError,
    #[case] expected: HierarchyErrorCode,
    #[case] expected_name: &'static str,
) {
    assert_code_contract!(error, expected, expected_name);
}

#[cfg(feature = "cpu")]
#[rstest]
#[case(HnswError::EmptyBuild, HnswErrorCode::EmptyBuild, "EMPTY_BUILD")]
#[case(
    HnswError::InvalidParameters {
        reason: "bad".into(),
    },
    HnswErrorCode::InvalidParameters,
    "INVALID_PARAMETERS"
)]
#[case(
    HnswError::DuplicateNode { node: 3 },
    HnswErrorCode::DuplicateNode,
    "DUPLICATE_NODE"
)]
#[case(HnswError::GraphEmpty, HnswErrorCode::GraphEmpty, "GRAPH_EMPTY")]
#[case(
    HnswError::GraphInvariantViolation {
        message: "oops".into(),
    },
    HnswErrorCode::GraphInvariantViolation,
    "GRAPH_INVARIANT_VIOLATION"
)]
#[case(
    HnswError::NonFiniteDistance { left: 0, right: 1 },
    HnswErrorCode::NonFiniteDistance,
    "NON_FINITE_DISTANCE"
)]
#[case(
    HnswError::LockPoisoned { resource: "graph" },
    HnswErrorCode::LockPoisoned,
    "LOCK_POISONED"
)]
#[case(
    HnswError::DataSource(DataSourceError::EmptyData),
    HnswErrorCode::DataSource,
    "DATA_SOURCE"
)]
fn returns_expected_hnsw_code(
    #[case] error: HnswError,
    #[case] expected: HnswErrorCode,
    #[case] expected_name: &'static str,
) {
    assert_code_contract!(error, expected, expected_name);
}
