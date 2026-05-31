//! Builder and session configuration tests for [`super::ClusteringSession`].
//!
//! These tests verify how `ChutoroBuilder` produces session configuration,
//! rejects unsupported construction choices, and maps HNSW allocation failures.
//! They exercise the boundary between builder policy and the session
//! constructor while reusing data-source fixtures from [`super::common`].

use std::{num::NonZeroUsize, sync::Arc};

use rstest::rstest;

use super::common::{SessionTestSource, session_builder};
use crate::{
    ChutoroBuilder, ChutoroError, ClusteringSession, ExecutionStrategy, HnswParams, SessionConfig,
    SessionRefreshPolicy,
};

#[test]
fn builder_defaults_include_session_configuration() {
    let builder = ChutoroBuilder::new();

    assert_eq!(builder.min_cluster_size(), 5);
    assert_eq!(builder.hnsw_params(), &HnswParams::default());
    assert_eq!(
        builder.session_refresh_policy(),
        &SessionRefreshPolicy::manual()
    );
}

#[test]
fn refresh_policy_threshold_can_be_set_and_cleared() {
    let threshold = NonZeroUsize::new(10).expect("threshold must be non-zero");
    let policy = SessionRefreshPolicy::manual().with_refresh_every_n(Some(threshold));

    assert_eq!(policy.refresh_every_n(), Some(threshold));

    let cleared = policy.with_refresh_every_n(None);
    assert_eq!(cleared.refresh_every_n(), None);
}

#[rstest]
#[case(
    HnswParams::new(8, 32).expect("params").with_rng_seed(17),
    SessionRefreshPolicy::manual()
)]
#[case(
    HnswParams::new(12, 48).expect("params").with_rng_seed(91),
    SessionRefreshPolicy::manual().with_refresh_every_n(NonZeroUsize::new(7))
)]
fn build_session_derives_config_from_builder(
    session_builder: ChutoroBuilder,
    #[case] hnsw_params: HnswParams,
    #[case] refresh_policy: SessionRefreshPolicy,
) {
    let source = Arc::new(SessionTestSource::with_len(4));
    let session = session_builder
        .with_min_cluster_size(9)
        .with_hnsw_params(hnsw_params.clone())
        .with_session_refresh_policy(refresh_policy)
        .build_session(source)
        .expect("session must build");

    assert_eq!(session.config().min_cluster_size().get(), 9);
    assert_eq!(session.config().hnsw_params(), &hnsw_params);
    assert_eq!(session.config().refresh_policy(), &refresh_policy);
}

#[rstest]
#[case(0)]
#[case(3)]
#[case(6)]
fn build_session_does_not_validate_current_source_len(
    session_builder: ChutoroBuilder,
    #[case] len: usize,
) {
    let source = Arc::new(SessionTestSource::with_len(len));
    let session = session_builder
        .with_min_cluster_size(5)
        .build_session(source)
        .expect("session creation must not validate current source length");

    assert_eq!(session.point_count(), 0);
    assert_eq!(session.snapshot_version(), 0);
}

#[rstest]
fn build_session_rejects_zero_min_cluster_size(session_builder: ChutoroBuilder) {
    let source = Arc::new(SessionTestSource::with_len(0));
    let err = session_builder
        .with_min_cluster_size(0)
        .build_session(source)
        .expect_err("zero min_cluster_size must fail");

    assert_eq!(err, ChutoroError::InvalidMinClusterSize { got: 0 });
}

#[rstest]
fn build_session_rejects_gpu_preferred_execution_strategy(session_builder: ChutoroBuilder) {
    let source = Arc::new(SessionTestSource::with_len(2));
    let err = session_builder
        .with_execution_strategy(ExecutionStrategy::GpuPreferred)
        .build_session(source)
        .expect_err("session creation must remain CPU-only");

    assert_eq!(
        err,
        ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred,
        }
    );
}

#[test]
fn build_session_maps_hnsw_construction_failure_to_cpu_hnsw_failure() {
    let source = Arc::new(SessionTestSource::with_len(0));
    let config = SessionConfig::new(
        NonZeroUsize::new(5).expect("minimum cluster size must be non-zero"),
        HnswParams::default(),
        SessionRefreshPolicy::manual(),
    );

    let err = ClusteringSession::new_failing_for_test(config, source)
        .expect_err("CpuHnsw construction failure must surface as CpuHnswFailure");
    assert!(
        matches!(err, ChutoroError::CpuHnswFailure { .. }),
        "expected CpuHnswFailure, got {err:?}"
    );
}
