//! Tests for the `Chutoro` orchestration API.

mod common;

use chutoro_core::{
    ChutoroBuilder, ChutoroError, ClusterId, ClusteringResult, DataSource, ExecutionStrategy,
};
use common::Dummy;
use rstest::{fixture, rstest};

#[fixture]
fn dummy() -> Dummy {
    Dummy::new(vec![1.0, 3.0, 6.0])
}

#[fixture]
fn small_dummy() -> Dummy {
    Dummy::new(vec![2.0, 5.0])
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
fn run_cpu_partitions_by_min_cluster_size() {
    let source = Dummy::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .build()
        .expect("configuration must be valid");
    let result = chutoro.run(&source).expect("run must succeed");
    let assignment_ids: Vec<u64> = result.assignments().iter().map(|id| id.get()).collect();
    assert_eq!(assignment_ids, vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(result.cluster_count(), 3);
}

#[rstest]
fn run_empty_source_errors() {
    let chutoro = ChutoroBuilder::new()
        .build()
        .expect("configuration must be valid");
    let empty = Dummy::new(vec![]);
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
fn builder_rejects_gpu_preferred() {
    let err = ChutoroBuilder::new()
        .with_execution_strategy(ExecutionStrategy::GpuPreferred)
        .build()
        .expect_err("builder must reject GPU preference");
    assert!(matches!(
        err,
        ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred,
        }
    ));
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
