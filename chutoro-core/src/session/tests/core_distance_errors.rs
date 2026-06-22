//! Error and invariant tests for session core-distance recomputation.

use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use rstest::rstest;

use super::common::{make_session, session_builder};
use crate::{
    ChutoroBuilder, ChutoroError, DataSource, DataSourceError, DistanceCacheConfig, HnswParams,
    MetricDescriptor,
};

#[rstest]
#[should_panic(expected = "assertion `left == right` failed")]
fn core_distance_asserts_storage_alignment(session_builder: ChutoroBuilder) {
    let (mut session, _) = make_session(session_builder, 1);

    session.append(&[0]).expect("append must succeed");
    session.dirty_core_distances.clear();

    let _ = session.core_distance(0);
}

#[test]
fn recompute_core_distances_propagates_data_source_errors() {
    let source = Arc::new(FailableSource::new(FailureMode::DataSource));
    let mut session = build_failable_session(Arc::clone(&source));

    session.append(&[0, 1, 2]).expect("append must succeed");
    source.fail();

    let err = session
        .recompute_core_distances_full()
        .expect_err("recompute must propagate data source failure");

    assert!(
        matches!(err, ChutoroError::DataSource { .. }),
        "expected data source error, got {err:?}"
    );
}

#[test]
fn recompute_core_distances_propagates_hnsw_errors() {
    let source = Arc::new(FailableSource::new(FailureMode::NonFinite));
    let mut session = build_failable_session(Arc::clone(&source));

    session.append(&[0, 1, 2]).expect("append must succeed");
    source.fail();

    let err = session
        .recompute_core_distances_full()
        .expect_err("recompute must propagate HNSW failure");

    assert!(
        matches!(err, ChutoroError::CpuHnswFailure { .. }),
        "expected HNSW error, got {err:?}"
    );
}

fn build_failable_session(source: Arc<FailableSource>) -> crate::ClusteringSession<FailableSource> {
    let cache_entries = NonZeroUsize::new(1).expect("cache size must be non-zero");
    let hnsw_params = HnswParams::new(2, 4)
        .expect("HNSW params must be valid")
        .with_distance_cache_config(
            DistanceCacheConfig::new(cache_entries).with_ttl(Some(Duration::ZERO)),
        );

    ChutoroBuilder::new()
        .with_min_cluster_size(1)
        .with_hnsw_params(hnsw_params)
        .build_session(source)
        .expect("session must build")
}

#[derive(Debug)]
struct FailableSource {
    values: Vec<f32>,
    should_fail: AtomicBool,
    mode: FailureMode,
}

impl FailableSource {
    fn new(mode: FailureMode) -> Self {
        Self {
            values: vec![0.0, 1.0, 2.0],
            should_fail: AtomicBool::new(false),
            mode,
        }
    }

    fn fail(&self) {
        self.should_fail.store(true, Ordering::SeqCst);
    }
}

impl DataSource for FailableSource {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn name(&self) -> &str {
        "failable-session-source"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        if self.should_fail.load(Ordering::SeqCst) {
            return match self.mode {
                FailureMode::DataSource => Err(DataSourceError::OutOfBounds { index: i.max(j) }),
                FailureMode::NonFinite => Ok(f32::NAN),
            };
        }

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
        MetricDescriptor::new("failable-session-source:abs")
    }
}

#[derive(Clone, Copy, Debug)]
enum FailureMode {
    DataSource,
    NonFinite,
}
