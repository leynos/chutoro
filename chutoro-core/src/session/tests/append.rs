//! Append semantics tests for clustering sessions.

use std::sync::Arc;

use rstest::rstest;

use super::common::{SessionTestSource, harvest_expected_edges, make_session, session_builder};
use crate::{ChutoroBuilder, ChutoroError, DataSource, DataSourceErrorCode, HnswParams};

#[rstest]
fn append_empty_slice_is_noop(session_builder: ChutoroBuilder) {
    let (mut session, _) = make_session(session_builder, 4);

    session.append(&[]).expect("empty append must succeed");

    assert_eq!(session.point_count(), 0);
    assert_eq!(session.snapshot_version(), 0);
    assert!(session.pending_edges.is_empty());
}

#[rstest]
fn append_single_index_increases_point_count(session_builder: ChutoroBuilder) {
    let (mut session, _) = make_session(session_builder, 4);

    session.append(&[0]).expect("first append must succeed");

    assert_eq!(session.point_count(), 1);
    assert_eq!(session.snapshot_version(), 0);
    assert!(session.pending_edges.is_empty());
}

#[rstest]
fn append_batch_accumulates_direct_harvested_edges(session_builder: ChutoroBuilder) {
    let hnsw_params = HnswParams::new(4, 16)
        .expect("HNSW params must be valid")
        .with_rng_seed(41);
    let source = Arc::new(SessionTestSource::with_len(6));
    let mut session = session_builder
        .with_hnsw_params(hnsw_params.clone())
        .build_session(Arc::clone(&source))
        .expect("session must build");
    let indices = [0, 1, 2, 3];
    let expected_edges = harvest_expected_edges(hnsw_params, source.as_ref(), &indices);

    session.append(&indices).expect("batch append must succeed");

    assert_eq!(session.point_count(), indices.len());
    assert_eq!(session.snapshot_version(), 0);
    assert_eq!(session.pending_edges, expected_edges);
}

#[rstest]
fn append_delegates_duplicate_rejection_to_hnsw(session_builder: ChutoroBuilder) {
    // Duplicate detection is delegated to CpuHnsw::insert_harvesting, which
    // returns HnswError::DuplicateIndex, mapped here to ChutoroError::CpuHnswFailure.
    let (mut session, _) = make_session(session_builder, 2);

    session.append(&[0]).expect("first append must succeed");
    let err = session
        .append(&[0])
        .expect_err("duplicate append must fail");

    assert!(
        matches!(err, ChutoroError::CpuHnswFailure { .. }),
        "expected CpuHnswFailure, got {err:?}"
    );
    assert_eq!(session.point_count(), 1);
}

#[rstest]
fn append_rejects_out_of_bounds_index(session_builder: ChutoroBuilder) {
    let (mut session, source) = make_session(session_builder, 2);

    let err = session
        .append(&[source.len()])
        .expect_err("out-of-bounds append must fail");

    assert!(
        matches!(err, ChutoroError::DataSource { .. }),
        "expected DataSource error, got {err:?}"
    );
    assert_eq!(
        err.data_source_code(),
        Some(DataSourceErrorCode::OutOfBounds)
    );
    assert_eq!(session.point_count(), 0);
}

#[rstest]
fn append_failure_preserves_prior_successes(session_builder: ChutoroBuilder) {
    // Use ≥3 source points so that inserting index 1 into a graph that
    // already contains index 0 produces at least one harvested edge.
    // We mirror append_batch_accumulates_direct_harvested_edges to establish
    // the expected edge set independently via CpuHnsw::insert_harvesting.
    let hnsw_params = HnswParams::new(4, 16)
        .expect("HNSW params must be valid")
        .with_rng_seed(41);
    let source = Arc::new(SessionTestSource::with_len(4));
    let mut session = session_builder
        .with_hnsw_params(hnsw_params.clone())
        .build_session(Arc::clone(&source))
        .expect("session must build");

    // Build the expected edge set using the direct index as a baseline.
    let expected_edges = harvest_expected_edges(hnsw_params, source.as_ref(), &[0, 1]);
    // expected_edges is non-empty at this point because inserting index 1
    // into a graph that already contains index 0 harvests their mutual edge.
    assert!(
        !expected_edges.is_empty(),
        "test precondition: at least one edge must be harvested from the first two insertions"
    );

    // Now append the two valid indices followed by one out-of-bounds index.
    let err = session
        .append(&[0, 1, source.len()])
        .expect_err("out-of-bounds index must cause failure");

    assert!(
        matches!(err, ChutoroError::DataSource { .. }),
        "expected DataSource error, got {err:?}"
    );
    assert_eq!(
        err.data_source_code(),
        Some(DataSourceErrorCode::OutOfBounds)
    );
    assert_eq!(
        session.point_count(),
        2,
        "the two successful insertions must be preserved"
    );
    assert_eq!(session.snapshot_version(), 0);
    assert_eq!(
        session.pending_edges, expected_edges,
        "harvested edges from successful insertions must survive a later failure"
    );
}

#[rstest]
fn append_does_not_publish_label_snapshot(session_builder: ChutoroBuilder) {
    let (mut session, _) = make_session(session_builder, 4);

    session.append(&[0, 1, 2]).expect("append must succeed");

    assert_eq!(session.snapshot_version(), 0);
}
