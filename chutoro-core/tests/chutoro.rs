//! Tests for the `Chutoro` orchestration API.

mod common;

use chutoro_core::{
    ChutoroBuilder, ChutoroError, ClusterId, ClusteringResult, DataSource, DataSourceError,
    ExecutionStrategy, NonContiguousClusterIds,
};
use common::Dummy;
use rstest::{fixture, rstest};
use std::sync::Arc;
use tracing::Level;
use tracing_subscriber::layer::SubscriberExt;

use chutoro_test_support::tracing::RecordingLayer;

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

    let chutoro = builder.clone().build().expect("defaults valid");
    assert_eq!(chutoro.min_cluster_size().get(), 5);
    assert_eq!(chutoro.execution_strategy(), ExecutionStrategy::Auto);
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

#[cfg(feature = "cpu")]
#[rstest]
fn run_records_core_tracing(dummy: Dummy) {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_execution_strategy(ExecutionStrategy::CpuOnly)
        .build()
        .expect("configuration must be valid");
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    let result = tracing::subscriber::with_default(subscriber, || chutoro.run(&dummy))
        .expect("run must succeed");
    assert_eq!(result.assignments().len(), dummy.len());

    let spans = layer.spans();
    assert!(spans.iter().any(|span| span.name == "core.run"));
    assert!(spans.iter().any(|span| span.name == "core.run_cpu"));
}

#[rstest]
fn run_logs_empty_source_warning() {
    let chutoro = ChutoroBuilder::new()
        .build()
        .expect("configuration must be valid");
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    let err = tracing::subscriber::with_default(subscriber, || chutoro.run(&Dummy::new(vec![])))
        .expect_err("empty sources must fail");
    assert!(matches!(err, ChutoroError::EmptySource { .. }));

    let spans = layer.spans();
    let run_span = spans
        .iter()
        .find(|span| span.name == "core.run")
        .expect("core.run span must exist");
    assert_eq!(run_span.fields.get("items"), Some(&"0".to_owned()));

    let events = layer.events();
    assert!(events.iter().any(|event| {
        event.level == Level::WARN
            && event
                .fields
                .get("message")
                .is_some_and(|value| value == "data source is empty, returning error")
    }));
}

#[cfg(not(feature = "cpu"))]
#[rstest]
#[case::auto(ExecutionStrategy::Auto)]
#[case::cpu_only(ExecutionStrategy::CpuOnly)]
fn run_cpu_strategies_error_without_cpu_feature(#[case] strategy: ExecutionStrategy, dummy: Dummy) {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(dummy.len())
        .with_execution_strategy(strategy)
        .build()
        .expect("configuration must be valid");
    let err = chutoro
        .run(&dummy)
        .expect_err("run must reject CPU strategies when the cpu feature is disabled");
    assert!(matches!(
        err,
        ChutoroError::BackendUnavailable { requested }
            if requested == strategy
    ));
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

#[cfg(not(feature = "gpu"))]
#[rstest]
fn builder_rejects_gpu_preferred_without_feature(dummy: Dummy) {
    let err = ChutoroBuilder::new()
        .with_min_cluster_size(dummy.len())
        .with_execution_strategy(ExecutionStrategy::GpuPreferred)
        .build()
        .expect_err("builder must reject GPU preference when feature is disabled");
    assert!(matches!(
        err,
        ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred,
        }
    ));
}

#[cfg(feature = "gpu")]
#[rstest]
fn run_gpu_preferred_errors_until_backend_available(dummy: Dummy) {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(dummy.len())
        .with_execution_strategy(ExecutionStrategy::GpuPreferred)
        .build()
        .expect("builder must allow runtime GPU preference");
    let err = chutoro
        .run(&dummy)
        .expect_err("gpu runs must fail until a backend is available");
    assert!(matches!(
        err,
        ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred
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

#[rstest]
#[case::missing_zero(vec![ClusterId::new(1)], NonContiguousClusterIds::MissingZero, "assignments must start at zero")]
#[case::gap(vec![ClusterId::new(0), ClusterId::new(2)], NonContiguousClusterIds::Gap, "assignments must be contiguous")]
#[case::overflow(vec![ClusterId::new(u64::MAX)], NonContiguousClusterIds::Overflow, "assignments must stay within usize range")]
#[case::duplicate(
    vec![ClusterId::new(0), ClusterId::new(0), ClusterId::new(2)],
    NonContiguousClusterIds::Duplicate,
    "assignments must not duplicate identifiers when clusters are missing",
)]
fn try_from_assignments_validates_contiguity(
    #[case] assignments: Vec<ClusterId>,
    #[case] expected_error: NonContiguousClusterIds,
    #[case] error_message: &str,
) {
    let err = ClusteringResult::try_from_assignments(assignments).expect_err(error_message);
    assert_eq!(err, expected_error);
}

#[test]
fn datasource_error_display_includes_index() {
    let err = DataSourceError::OutOfBounds { index: 5 };
    assert_eq!(format!("{err}"), "index 5 is out of bounds");
}

#[test]
fn chutoro_error_datasource_includes_source_name() {
    let inner = DataSourceError::DimensionMismatch { left: 2, right: 3 };
    let err = ChutoroError::DataSource {
        data_source: Arc::from("dummy"),
        error: inner.clone(),
    };
    assert!(matches!(
        err,
        ChutoroError::DataSource { ref data_source, ref error }
            if data_source.as_ref() == "dummy" && error == &inner
    ));
    assert!(format!("{err}").contains("dummy"));
}
