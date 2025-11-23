//! Metadata-oriented tests covering error codes and reporting.

use crate::{
    DataSourceError,
    hnsw::{HnswError, HnswErrorCode},
};

#[test]
fn exposes_machine_readable_error_codes() {
    assert_eq!(HnswError::EmptyBuild.code(), HnswErrorCode::EmptyBuild);
    assert_eq!(
        HnswError::InvalidParameters {
            reason: "bad".into(),
        }
        .code(),
        HnswErrorCode::InvalidParameters,
    );
    assert_eq!(
        HnswError::DuplicateNode { node: 3 }.code(),
        HnswErrorCode::DuplicateNode,
    );
    assert_eq!(HnswError::GraphEmpty.code(), HnswErrorCode::GraphEmpty);
    assert_eq!(
        HnswError::GraphInvariantViolation {
            message: "oops".into(),
        }
        .code(),
        HnswErrorCode::GraphInvariantViolation,
    );
    assert_eq!(
        HnswError::NonFiniteDistance { left: 0, right: 1 }.code(),
        HnswErrorCode::NonFiniteDistance,
    );
    assert_eq!(
        HnswError::LockPoisoned { resource: "graph" }.code(),
        HnswErrorCode::LockPoisoned,
    );
    assert_eq!(
        HnswError::from(DataSourceError::EmptyData).code(),
        HnswErrorCode::DataSource,
    );
    assert_eq!(HnswErrorCode::DataSource.as_str(), "DATA_SOURCE");
    assert_eq!(HnswErrorCode::LockPoisoned.as_str(), "LOCK_POISONED");
}
