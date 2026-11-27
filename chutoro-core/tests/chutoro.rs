//! Integration tests for the `Chutoro` orchestration API.
//!
//! The root builds clusterings end to end and reads the session and pipeline
//! facades back. Assignment-validation examples sit in the private
//! `assignment_validation` child module, and the HNSW-parameter and memory-limit
//! cases sit in `hnsw_params`. This root declares both with `#[path]` so those
//! cases still run in one integration-test binary while each file stays within
//! the module-size budget.

#[path = "chutoro/assignment_validation.rs"]
mod assignment_validation;
mod common;
#[path = "support/failable_source.rs"]
mod failable_source;
#[cfg(feature = "cpu")]
#[path = "chutoro/hnsw_params.rs"]
mod hnsw_params;

use chutoro_core::{
    ChutoroBuilder, ChutoroError, DataSource, DataSourceError, ExecutionStrategy, MetricDescriptor,
};
use common::Dummy;
use failable_source::{FailableSource, FailureMode};
use rstest::{fixture, rstest};
use std::sync::Arc;
use tracing::Level;
use tracing_subscriber::layer::SubscriberExt;

use chutoro_test_support::tracing::{RecordingLayer, pin_callsite_interest};

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
    let _pin = pin_callsite_interest();
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
    let _pin = pin_callsite_interest();
    let chutoro = ChutoroBuilder::new()
        .build()
        .expect("configuration must be valid");
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    let result = tracing::subscriber::with_default(subscriber, || chutoro.run(&Dummy::new(vec![])));
    let spans = layer.spans();
    let events = layer.events();

    let err = result.expect_err("empty sources must fail");
    assert!(matches!(err, ChutoroError::EmptySource { .. }));

    let run_span = spans
        .iter()
        .find(|span| span.name == "core.run")
        .unwrap_or_else(|| {
            panic!("core.run span must exist; spans: {spans:?}, events: {events:?}")
        });
    assert_eq!(run_span.fields.get("items"), Some(&"0".to_owned()));

    assert!(
        events.iter().any(|event| {
            event.level == Level::WARN
                && event
                    .fields
                    .get("message")
                    .is_some_and(|value| value == "data source is empty, returning error")
        }),
        "the empty-source warning must be recorded; spans: {spans:?}, events: {events:?}"
    );
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

/// Verifies that one-shot runs preserve the public HNSW error split.
#[cfg(feature = "cpu")]
#[rstest]
#[case::data_source(FailureMode::DataSource)]
#[case::hnsw(FailureMode::NonFinite)]
#[case::pair_data_source(FailureMode::PairDataSource { left: 0, right: 1 })]
fn run_maps_hnsw_errors(#[case] mode: FailureMode) {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_execution_strategy(ExecutionStrategy::CpuOnly)
        .build()
        .expect("configuration must be valid");
    let source = FailableSource::new(mode);
    source.fail();

    let err = chutoro
        .run(&source)
        .expect_err("one-shot run must propagate the injected HNSW failure");
    source.recover();
    assert!(
        source.distance(0, 1).is_ok(),
        "recovery must restore fixture distance queries"
    );

    match mode {
        FailureMode::DataSource | FailureMode::PairDataSource { .. } => assert!(
            matches!(err, ChutoroError::DataSource { .. }),
            "expected data source error, got {err:?}"
        ),
        FailureMode::NonFinite => assert!(
            matches!(err, ChutoroError::CpuHnswFailure { .. }),
            "expected HNSW error, got {err:?}"
        ),
    }
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
