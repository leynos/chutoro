//! Tests for the public incremental session scaffolding.

use std::{num::NonZeroUsize, sync::Arc};

use rstest::rstest;

use crate::{
    ChutoroBuilder, ChutoroError, DataSource, DataSourceError, ExecutionStrategy, HnswParams,
    MetricDescriptor, SessionRefreshPolicy,
};

#[derive(Clone, Debug)]
struct SessionTestSource {
    values: Vec<f32>,
    name: &'static str,
}

impl SessionTestSource {
    fn with_len(len: usize) -> Self {
        Self {
            values: (0..len).map(|value| value as f32).collect(),
            name: "session-test",
        }
    }
}

impl DataSource for SessionTestSource {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn name(&self) -> &str {
        self.name
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let left = self
            .values
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = self
            .values
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((left - right).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("session-test:abs")
    }
}

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
    #[case] hnsw_params: HnswParams,
    #[case] refresh_policy: SessionRefreshPolicy,
) {
    let source = Arc::new(SessionTestSource::with_len(4));
    let session = ChutoroBuilder::new()
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
fn build_session_accepts_empty_and_undersized_sources(#[case] len: usize) {
    let source = Arc::new(SessionTestSource::with_len(len));
    let session = ChutoroBuilder::new()
        .with_min_cluster_size(5)
        .build_session(source)
        .expect("session creation must not validate current source length");

    assert_eq!(session.point_count(), 0);
    assert_eq!(session.snapshot_version(), 0);
}

#[test]
fn build_session_rejects_zero_min_cluster_size() {
    let source = Arc::new(SessionTestSource::with_len(0));
    let err = ChutoroBuilder::new()
        .with_min_cluster_size(0)
        .build_session(source)
        .expect_err("zero min_cluster_size must fail");

    assert_eq!(err, ChutoroError::InvalidMinClusterSize { got: 0 });
}

#[test]
fn build_session_rejects_gpu_preferred_execution_strategy() {
    let source = Arc::new(SessionTestSource::with_len(2));
    let err = ChutoroBuilder::new()
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
fn build_session_initialises_empty_state() {
    let source = Arc::new(SessionTestSource::with_len(6));
    let session = ChutoroBuilder::new()
        .build_session(Arc::clone(&source))
        .expect("session must build");

    assert_eq!(session.point_count(), 0);
    assert_eq!(session.snapshot_version(), 0);
    assert!(session.index.is_empty());
    assert!(session._core_distances.is_empty());
    assert!(session._mst_edges.is_empty());
    assert!(session._historical_edges.is_empty());
    assert!(session._pending_edges.is_empty());
    assert!(session._labels.is_empty());
    assert_eq!(session._last_refresh_len, 0);
    assert_eq!(Arc::strong_count(&session._labels), 1);
    assert!(Arc::ptr_eq(&session._source, &source));
}
