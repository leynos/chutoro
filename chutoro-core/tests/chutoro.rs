#![expect(clippy::expect_used, reason = "tests require contextual panics")]
//! Tests for the `Chutoro` orchestration API.

use chutoro_core::{
    ChutoroBuilder, ChutoroError, ClusterId, ClusteringResult, DataSource, DataSourceError,
    ExecutionStrategy,
};
use rstest::{fixture, rstest};

#[derive(Clone)]
struct Dummy(Vec<f32>);

#[fixture]
fn dummy() -> Dummy {
    Dummy(vec![1.0, 3.0, 6.0])
}

#[fixture]
fn small_dummy() -> Dummy {
    Dummy(vec![2.0, 5.0])
}

impl DataSource for Dummy {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn name(&self) -> &str {
        "dummy"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self
            .0
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self
            .0
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
}

#[rstest]
fn builder_defaults() {
    let builder = ChutoroBuilder::new();
    assert_eq!(builder.min_cluster_size(), 5);
    assert_eq!(builder.execution_strategy(), ExecutionStrategy::Auto);
}

#[rstest]
fn builder_rejects_zero_min_cluster_size() {
    let err = ChutoroBuilder::new()
        .with_min_cluster_size(0)
        .build()
        .expect_err("builder must reject zero min_cluster_size");
    assert!(matches!(
        err,
        ChutoroError::InvalidMinClusterSize { got: 0 }
    ));
}

#[rstest]
#[case::auto(ExecutionStrategy::Auto)]
#[case::cpu_only(ExecutionStrategy::CpuOnly)]
fn run_cpu_single_cluster(#[case] strategy: ExecutionStrategy, dummy: Dummy) {
    let len = dummy.len();
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(len)
        .with_execution_strategy(strategy)
        .build()
        .expect("configuration must be valid");
    let result = chutoro.run(&dummy).expect("run must succeed");
    assert_eq!(result.assignments().len(), dummy.len());
    assert_eq!(result.cluster_count(), 1);
    assert!(result.assignments().iter().all(|id| id.get() == 0));
}

#[rstest]
fn run_empty_source_errors() {
    let chutoro = ChutoroBuilder::new()
        .build()
        .expect("configuration must be valid");
    let empty = Dummy(vec![]);
    let err = chutoro
        .run(&empty)
        .expect_err("run must reject empty data sources");
    assert!(matches!(err, ChutoroError::EmptySource { .. }));
}

#[rstest]
fn run_insufficient_items_errors(small_dummy: Dummy) {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(4)
        .build()
        .expect("configuration must be valid");
    let err = chutoro
        .run(&small_dummy)
        .expect_err("run must enforce min_cluster_size");
    assert!(matches!(
        err,
        ChutoroError::InsufficientItems {
            items: 2,
            min_cluster_size,
            ..
        } if min_cluster_size.get() == 4
    ));
}

#[rstest]
fn run_gpu_preferred_errors(dummy: Dummy) {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(1)
        .with_execution_strategy(ExecutionStrategy::GpuPreferred)
        .build()
        .expect("configuration must be valid");
    let err = chutoro
        .run(&dummy)
        .expect_err("GPU preference must fail without a backend");
    assert!(matches!(err, ChutoroError::BackendUnavailable { .. }));
}

#[rstest]
#[case::single(vec![ClusterId::new(0)], 1)]
#[case::two_clusters(vec![ClusterId::new(0), ClusterId::new(1)], 2)]
fn cluster_count_matches_unique_assignments(
    #[case] assignments: Vec<ClusterId>,
    #[case] expected: usize,
) {
    let result = ClusteringResult::from_assignments(assignments);
    assert_eq!(result.cluster_count(), expected);
}
