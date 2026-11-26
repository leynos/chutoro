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

/// One recorded metric: how many samples it carries and their values.
struct Recorded {
    /// Number of samples, which is 1 for a counter.
    count: usize,
    /// Recorded values, in order. A gauge contributes none.
    values: Vec<f64>,
}

/// Collects the recorded metrics keyed by metric name and labels.
fn recorded_metrics(
    snapshot: metrics_util::debugging::Snapshot,
) -> Vec<(String, String, Recorded)> {
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
            let recorded = match value {
                DebugValue::Histogram(samples) => Recorded {
                    count: samples.len(),
                    values: samples.into_iter().map(|sample| sample.0).collect(),
                },
                DebugValue::Counter(count) => Recorded {
                    count: usize::try_from(count).unwrap_or(usize::MAX),
                    values: Vec::new(),
                },
                DebugValue::Gauge(_) => Recorded {
                    count: 0,
                    values: Vec::new(),
                },
            };
            (key.key().name().to_owned(), labels, recorded)
        })
        .collect()
}

/// Returns the recorded sample counts keyed by metric name and labels.
fn recorded_counts(snapshot: metrics_util::debugging::Snapshot) -> Vec<(String, String, usize)> {
    recorded_metrics(snapshot)
        .into_iter()
        .map(|(name, labels, recorded)| (name, labels, recorded.count))
        .collect()
}

/// Returns every value recorded for one histogram metric.
fn values_for(records: &[(String, String, Recorded)], name: &str) -> Vec<f64> {
    records
        .iter()
        .filter(|(metric, _, _)| metric == name)
        .flat_map(|(_, _, recorded)| recorded.values.iter().copied())
        .collect()
}

/// Converts a small count to the `f64` a histogram sample carries.
///
/// Refresh metrics count points, edges and clusters, so the `u32` conversion
/// cannot truncate in practice; a wider value would mean the counter overflowed.
fn as_sample(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// Returns the recorded count for one metric name, summing its label variants.
fn count_for(records: &[(String, String, usize)], name: &str) -> usize {
    records
        .iter()
        .filter(|(metric, _, _)| metric == name)
        .map(|(_, _, count)| *count)
        .sum()
}

/// Returns the recorded sample count for one metric of the richer type.
fn recorded_count_for(records: &[(String, String, Recorded)], name: &str) -> usize {
    records
        .iter()
        .filter(|(metric, _, _)| metric == name)
        .map(|(_, _, recorded)| recorded.count)
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

/// The recorded sample must describe the snapshot the refresh published.
///
/// Counting samples alone would pass a metric that recorded the wrong number,
/// so this case compares the recorded `cluster_count` value with the distinct
/// labels of the published snapshot and the recorded `candidate_edges` value
/// with the harvested-edge counter for the same append.
#[rstest]
fn refresh_metric_values_describe_the_published_snapshot(session_builder: ChutoroBuilder) {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let (records, published) = metrics::with_local_recorder(&recorder, || {
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
        (
            recorded_metrics(snapshotter.snapshot()),
            session.labels().to_vec(),
        )
    });

    let distinct = published
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert_eq!(
        values_for(&records, "chutoro.session.refresh.cluster_count"),
        vec![as_sample(distinct)],
        "cluster_count must record the published partition's distinct labels"
    );

    let harvested = recorded_count_for(&records, "chutoro.session.harvested_edges");
    assert_eq!(
        values_for(&records, "chutoro.session.refresh.candidate_edges"),
        vec![as_sample(harvested)],
        "candidate_edges must record every edge the append harvested and the merge saw"
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
