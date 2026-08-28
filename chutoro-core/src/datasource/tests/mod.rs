//! Tests for `DataSource` default batch behaviour.

use super::*;
use crate::test_utils::CountingSource;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

mod batch_first_source;

#[test]
fn batch_distances_invokes_scalar_distance() {
    let calls = Arc::new(AtomicUsize::new(0));
    let source = CountingSource::new(vec![0.0, 1.0, 3.0], Arc::clone(&calls));

    let distances = source
        .batch_distances(0, &[1, 2])
        .expect("batch distances should succeed");

    assert_eq!(distances, vec![1.0, 3.0]);
    assert_eq!(source.calls().load(Ordering::Relaxed), 2);
}

#[test]
fn batch_distances_propagates_errors() {
    let calls = Arc::new(AtomicUsize::new(0));
    let source = CountingSource::new(vec![0.0, 1.0], calls);

    let err = source
        .batch_distances(0, &[1, 5])
        .expect_err("invalid candidate must fail");

    assert!(
        matches!(err, DataSourceError::OutOfBounds { index: 5 }),
        "expected OutOfBounds with index 5, got {err:?}",
    );
}

#[test]
fn batch_distances_rejects_out_of_bounds_query_with_empty_candidates() {
    let calls = Arc::new(AtomicUsize::new(0));
    let source = CountingSource::new(vec![0.0, 1.0], Arc::clone(&calls));

    let err = source
        .batch_distances(usize::MAX, &[])
        .expect_err("invalid query must fail even for empty candidate sets");

    assert!(
        matches!(err, DataSourceError::OutOfBounds { index } if index == usize::MAX),
        "expected OutOfBounds with usize::MAX, got {err:?}",
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "query validation should fail before scalar distances are computed",
    );
}
