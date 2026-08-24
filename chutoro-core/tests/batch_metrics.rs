//! Metrics tests for one-shot batch execution.

#![cfg(feature = "metrics")]

mod common;

use metrics_util::debugging::DebugValue;
#[cfg(feature = "cpu")]
use tracing_subscriber::layer::SubscriberExt;

#[cfg(feature = "cpu")]
use chutoro_core::DataSource;
use chutoro_core::{ChutoroBuilder, ChutoroError, ExecutionStrategy};
#[cfg(feature = "cpu")]
use chutoro_test_support::tracing::RecordingLayer;

use common::Dummy;

const RUNS_TOTAL: &str = "chutoro.batch.runs_total";
#[cfg(feature = "cpu")]
const MAX_CONNECTIONS: &str = "chutoro.batch.max_connections";
#[cfg(feature = "cpu")]
const EFFECTIVE_EF_CONSTRUCTION: &str = "chutoro.batch.effective_ef_construction";
#[cfg(feature = "cpu")]
const ESTIMATED_BYTES: &str = "chutoro.batch.estimated_bytes";
#[cfg(feature = "cpu")]
const MEMORY_LIMIT_BYTES: &str = "chutoro.batch.memory_limit_bytes";

macro_rules! metric_value {
    ($snapshot:expr, $name:expr, $labels:expr $(,)?) => {
        $snapshot
            .iter()
            .find(|(key, _, _, _)| {
                key.key().name() == $name
                    && key.key().labels().count() == $labels.len()
                    && $labels.iter().all(|(expected_key, expected_value)| {
                        key.key().labels().any(|label| {
                            label.key() == *expected_key && label.value() == *expected_value
                        })
                    })
            })
            .map(|(_, _, _, value)| value)
            .expect("metric with the expected bounded labels must be recorded")
    };
}

#[cfg(feature = "cpu")]
fn assert_histogram_sample(value: &DebugValue, expected: f64) {
    let DebugValue::Histogram(samples) = value else {
        panic!("expected a Histogram metric value, got {value:?}");
    };
    assert!(
        samples.iter().any(|sample| sample.into_inner() == expected),
        "expected histogram sample {expected}, got {samples:?}"
    );
}

macro_rules! assert_outcome {
    ($snapshot:expr, $backend:expr, $outcome:expr, $error_code:expr $(,)?) => {
        assert_eq!(
            metric_value!(
                $snapshot,
                RUNS_TOTAL,
                &[
                    ("backend", $backend),
                    ("outcome", $outcome),
                    ("error_code", $error_code),
                ],
            ),
            &DebugValue::Counter(1),
        );
    };
}

#[cfg(feature = "cpu")]
#[test]
fn successful_cpu_run_records_bounded_resources_and_tracing() {
    use chutoro_core::{HnswParams, estimate_peak_bytes_for_hnsw_params};

    let params = HnswParams::new(2, 4).expect("parameters must be valid");
    let source = Dummy::new(vec![1.0, 3.0, 6.0, 10.0]);
    let estimate = estimate_peak_bytes_for_hnsw_params(source.len(), &params);
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_execution_strategy(ExecutionStrategy::CpuOnly)
        .with_hnsw_params(params.clone())
        .build()
        .expect("configuration must be valid");
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    metrics::with_local_recorder(&recorder, || {
        tracing::subscriber::with_default(subscriber, || {
            chutoro.run(&source).expect("CPU batch run must succeed");
        });
    });

    let snapshot = snapshotter.snapshot().into_vec();
    assert_outcome!(&snapshot, "cpu", "success", "none");
    assert_histogram_sample(
        metric_value!(&snapshot, MAX_CONNECTIONS, &[("backend", "cpu")]),
        2.0,
    );
    assert_histogram_sample(
        metric_value!(&snapshot, EFFECTIVE_EF_CONSTRUCTION, &[("backend", "cpu")],),
        4.0,
    );
    assert_histogram_sample(
        metric_value!(&snapshot, ESTIMATED_BYTES, &[("backend", "cpu")]),
        estimate as f64,
    );

    let run_span = layer
        .spans()
        .into_iter()
        .find(|span| span.name == "core.run")
        .expect("batch run span must be recorded");
    assert_eq!(run_span.fields.get("backend"), Some(&"cpu".to_owned()));
    assert!(!run_span.fields.contains_key("data_source"));
}

#[cfg(feature = "cpu")]
#[test]
fn memory_limit_rejection_records_bounded_metrics_and_tracing() {
    use chutoro_core::{HnswParams, estimate_peak_bytes_for_hnsw_params};

    let params = HnswParams::new(2, 4).expect("parameters must be valid");
    let source = Dummy::new(vec![1.0, 3.0, 6.0, 10.0]);
    let estimate = estimate_peak_bytes_for_hnsw_params(source.len(), &params);
    let limit = estimate - 1;
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_execution_strategy(ExecutionStrategy::CpuOnly)
        .with_hnsw_params(params)
        .with_max_bytes(limit)
        .build()
        .expect("configuration must be valid");
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    metrics::with_local_recorder(&recorder, || {
        let error = tracing::subscriber::with_default(subscriber, || chutoro.run(&source))
            .expect_err("one byte below the estimate must reject the run");
        assert!(matches!(error, ChutoroError::MemoryLimitExceeded { .. }));
    });

    let snapshot = snapshotter.snapshot().into_vec();
    assert_outcome!(&snapshot, "cpu", "error", "CHUTORO_MEMORY_LIMIT_EXCEEDED",);
    assert_histogram_sample(
        metric_value!(&snapshot, MAX_CONNECTIONS, &[("backend", "cpu")]),
        2.0,
    );
    assert_histogram_sample(
        metric_value!(&snapshot, EFFECTIVE_EF_CONSTRUCTION, &[("backend", "cpu")],),
        4.0,
    );
    assert_histogram_sample(
        metric_value!(&snapshot, ESTIMATED_BYTES, &[("backend", "cpu")]),
        estimate as f64,
    );
    assert_histogram_sample(
        metric_value!(&snapshot, MEMORY_LIMIT_BYTES, &[("backend", "cpu")],),
        limit as f64,
    );
    let event = layer
        .events()
        .into_iter()
        .find(|event| {
            event
                .fields
                .get("message")
                .is_some_and(|message| message == "CPU memory estimate exceeds configured limit")
        })
        .expect("memory limit rejection event must be recorded");
    assert_eq!(
        event.fields.get("error_code"),
        Some(&"CHUTORO_MEMORY_LIMIT_EXCEEDED".to_owned())
    );
    assert!(!event.fields.contains_key("data_source"));
}

#[test]
fn empty_and_insufficient_rejections_record_stable_outcomes() {
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        let empty_error = ChutoroBuilder::new()
            .build()
            .expect("configuration must be valid")
            .run(&Dummy::new(Vec::new()))
            .expect_err("empty source must be rejected");
        assert!(matches!(empty_error, ChutoroError::EmptySource { .. }));

        let insufficient_error = ChutoroBuilder::new()
            .with_min_cluster_size(3)
            .build()
            .expect("configuration must be valid")
            .run(&Dummy::new(vec![1.0, 2.0]))
            .expect_err("undersized source must be rejected");
        assert!(matches!(
            insufficient_error,
            ChutoroError::InsufficientItems { .. }
        ));
    });

    let backend = if cfg!(feature = "cpu") {
        "cpu"
    } else {
        "unavailable"
    };
    let snapshot = snapshotter.snapshot().into_vec();
    assert_outcome!(&snapshot, backend, "error", "CHUTORO_EMPTY_SOURCE",);
    assert_outcome!(&snapshot, backend, "error", "CHUTORO_INSUFFICIENT_ITEMS",);
}

#[cfg(not(feature = "cpu"))]
#[test]
fn unavailable_cpu_backend_records_stable_outcome() {
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        let error = ChutoroBuilder::new()
            .with_min_cluster_size(2)
            .with_execution_strategy(ExecutionStrategy::CpuOnly)
            .build()
            .expect("configuration must be valid")
            .run(&Dummy::new(vec![1.0, 2.0]))
            .expect_err("unavailable CPU backend must be rejected");
        assert!(matches!(error, ChutoroError::BackendUnavailable { .. }));
    });

    let snapshot = snapshotter.snapshot().into_vec();
    assert_outcome!(
        &snapshot,
        "unavailable",
        "error",
        "CHUTORO_BACKEND_UNAVAILABLE",
    );
}
