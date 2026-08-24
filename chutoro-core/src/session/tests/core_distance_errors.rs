//! Error and invariant tests for session core-distance recomputation.

use std::{
    error::Error,
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
    let (mut session, _) = make_session(session_builder, 1).expect("session must build");

    session.append(&[0]).expect("append must succeed");
    session.dirty_core_distances.clear();

    let _alignment_check = session.core_distance(0);
}

#[rstest]
#[case(FailureMode::DataSource, "data source failure")]
#[case(FailureMode::NonFinite, "HNSW failure")]
fn recompute_core_distances_propagates_errors(
    #[case] mode: FailureMode,
    #[case] failure_description: &str,
) {
    let source = Arc::new(FailableSource::new(mode));
    let mut session = build_failable_session(Arc::clone(&source)).expect("session must build");

    session.append(&[0, 1, 2]).expect("append must succeed");
    source.fail();

    let err = session
        .recompute_core_distances_full()
        .expect_err(&format!("recompute must propagate {failure_description}"));

    assert!(
        matches!(mode, FailureMode::DataSource | FailureMode::NonFinite),
        "pair data source failures belong to the dirty-state retention test",
    );
    if matches!(mode, FailureMode::DataSource) {
        assert!(
            matches!(err, ChutoroError::DataSource { .. }),
            "expected data source error, got {err:?}"
        );
    } else {
        assert!(
            matches!(err, ChutoroError::CpuHnswFailure { .. }),
            "expected HNSW error, got {err:?}"
        );
    }
}

#[test]
fn recompute_core_distances_keeps_new_points_dirty_when_existing_search_fails() {
    let source = Arc::new(FailableSource::new(FailureMode::PairDataSource {
        left: 0,
        right: 2,
    }));
    let mut session = build_failable_session(Arc::clone(&source)).expect("session must build");

    session.append(&[0, 2]).expect("first append must succeed");
    session
        .recompute_core_distances_full()
        .expect("first recompute must succeed");
    session.append(&[1]).expect("second append must succeed");

    source.fail();
    let err = session
        .recompute_core_distances()
        .expect_err("touched existing search must fail");

    assert!(
        matches!(err, ChutoroError::DataSource { .. }),
        "expected data source error, got {err:?}"
    );
    assert_eq!(
        session.core_distance(1),
        None,
        "new point must remain dirty so retry recomputes touched existing points"
    );

    source.recover();
    session
        .recompute_core_distances()
        .expect("retry must recompute the still-dirty new point");

    assert!(session.core_distance(1).is_some());
}

fn build_failable_session(
    source: Arc<FailableSource>,
) -> Result<crate::ClusteringSession<FailableSource>, Box<dyn Error>> {
    let cache_entries = NonZeroUsize::new(1).ok_or("cache size must be non-zero")?;
    let hnsw_params = HnswParams::new(2, 4)?.with_distance_cache_config(
        DistanceCacheConfig::new(cache_entries).with_ttl(Some(Duration::ZERO)),
    );

    Ok(ChutoroBuilder::new()
        .with_min_cluster_size(1)
        .with_hnsw_params(hnsw_params)
        .build_session(source)?)
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

    fn recover(&self) {
        self.should_fail.store(false, Ordering::SeqCst);
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
                FailureMode::PairDataSource { left, right } if is_pair(i, j, left, right) => {
                    Err(DataSourceError::OutOfBounds { index: i.max(j) })
                }
                FailureMode::PairDataSource { .. } => Ok((self.values[i] - self.values[j]).abs()),
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
    PairDataSource { left: usize, right: usize },
}

fn is_pair(i: usize, j: usize, left: usize, right: usize) -> bool {
    (i == left && j == right) || (i == right && j == left)
}
