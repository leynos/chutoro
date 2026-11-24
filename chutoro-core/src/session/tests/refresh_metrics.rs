//! Metrics tests for [`ClusteringSession::refresh`].
//!
//! Compiled only under `#[cfg(feature = "metrics")]`. Each case installs a
//! [`DebuggingRecorder`] locally, runs one refresh through the public API, and
//! asserts the exact metric surface: the four documented refresh metrics and
//! the bounded failure-reason label. Assertions read the recorded values, so a
//! metric that stops being emitted fails the test rather than passing
//! silently.

use std::sync::Arc;

use metrics_util::debugging::{DebugValue, DebuggingRecorder};
use rstest::rstest;

use super::common::{make_session, session_builder};
use super::failable_source::{FailableSource, FailureMode};
use crate::{ChutoroBuilder, HnswParams, MetricDescriptor, test_utils::NumericTestSource};

/// Collects the recorded sample counts keyed by metric name and labels.
fn recorded_counts(snapshot: metrics_util::debugging::Snapshot) -> Vec<(String, String, usize)> {
    snapshot
        .into_hashmap()
        .into_iter()
        .map(|(key, (_, _, value))| {
            let labels = key
                .key()
                .labels()
                .map(|label| format!("{}={}", label.key(), label.value()))
                .collect::<Vec<_>>()
                .join(",");
            // Only histograms and counters are asserted here; other value
            // types cannot occur for the refresh metrics.
            let count = match value {
                DebugValue::Histogram(samples) => samples.len(),
                DebugValue::Counter(count) => usize::try_from(count).unwrap_or(usize::MAX),
                DebugValue::Gauge(_) => 0,
            };
            (key.key().name().to_owned(), labels, count)
        })
        .collect()
}

/// Returns the recorded count for one metric name, summing its label variants.
fn count_for(records: &[(String, String, usize)], name: &str) -> usize {
    records
        .iter()
        .filter(|(metric, _, _)| metric == name)
        .map(|(_, _, count)| *count)
        .sum()
}

#[rstest]
fn refresh_records_its_metrics(session_builder: ChutoroBuilder) {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let records = metrics::with_local_recorder(&recorder, || {
        let source = Arc::new(
            NumericTestSource::new("refresh-metrics", vec![0.0, 1.0, 2.0, 30.0, 31.0, 32.0])
                .using_metric_descriptor(MetricDescriptor::new("refresh-metrics:abs")),
        );
        let mut session = session_builder
            .with_min_cluster_size(2)
            .build_session(source)
            .expect("session must build");
        session
            .append(&[0, 1, 2, 3, 4, 5])
            .expect("append must succeed");
        session.refresh().expect("refresh must succeed");
        recorded_counts(snapshotter.snapshot())
    });

    assert_eq!(
        count_for(&records, "chutoro.session.refresh.seconds"),
        1,
        "one duration sample per refresh, got {records:?}"
    );
    assert_eq!(
        count_for(&records, "chutoro.session.refresh.candidate_edges"),
        1,
        "one merged-edge sample per refresh, got {records:?}"
    );
    assert_eq!(
        count_for(&records, "chutoro.session.refresh.cluster_count"),
        1,
        "one distinct-label sample per refresh, got {records:?}"
    );
    assert_eq!(
        count_for(&records, "chutoro.session.refresh.errors_total"),
        0,
        "a successful refresh must not count an error, got {records:?}"
    );
}

/// The two error classes a caller can actually reach are labelled distinctly.
///
/// A non-contiguous append fails in the rebuild (the caller-input MST path),
/// and a failing data source fails during the core-distance recompute. Both
/// must land on `errors_total` under their own reason, so a dashboard can tell
/// a caller mistake from a provider fault.
#[rstest]
fn refresh_error_is_labelled_by_reason() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let records = metrics::with_local_recorder(&recorder, || {
        let (mut mst_session, _) = make_session(ChutoroBuilder::new().with_min_cluster_size(2), 4)
            .expect("session must build");
        // Skipping index 2 makes the rebuild reject the session.
        mst_session
            .append(&[0, 1, 3])
            .expect("a non-contiguous append is accepted");
        mst_session.refresh().expect_err("the refresh must fail");

        let source = std::sync::Arc::new(FailableSource::new(FailureMode::DataSource));
        let mut data_session = ChutoroBuilder::new()
            .with_min_cluster_size(2)
            .with_hnsw_params(HnswParams::new(2, 4).expect("test parameters must be valid"))
            .build_session(std::sync::Arc::clone(&source))
            .expect("session must build");
        data_session
            .append(&[0, 1])
            .expect("append must succeed before the failure");
        source.fail();
        data_session.refresh().expect_err("the refresh must fail");

        recorded_counts(snapshotter.snapshot())
    });

    for reason in ["mst_failure", "data_source"] {
        let labelled = records
            .iter()
            .find(|(metric, labels, _)| {
                metric == "chutoro.session.refresh.errors_total" && labels.contains(reason)
            })
            .unwrap_or_else(|| {
                panic!(
                    "the {reason} failure must be counted with its reason label, got {records:?}"
                )
            });
        assert_eq!(labelled.2, 1, "each reason must be counted once");
    }
}
