//! Metric assertions for the trim-scoring boundary.
//!
//! These drive [`CpuHnsw::score_trim_jobs_with_metrics`], the boundary the
//! insertion path calls, so the scheduling test in the parent module stays
//! free of recorder setup and the instruments are pinned for the empty,
//! successful and failed batch paths.

use std::collections::HashMap;

use metrics_util::debugging::{DebugValue, DebuggingRecorder};

use super::*;

/// Runs `score` under a local recorder and returns its value with the
/// metrics it emitted, keyed by metric name.
fn record_trim_metrics<R>(score: impl FnOnce() -> R) -> (R, HashMap<String, DebugValue>) {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let value = ::metrics::with_local_recorder(&recorder, score);
    let metrics_by_name = snapshotter
        .snapshot()
        .into_hashmap()
        .into_iter()
        .map(|(key, (_, _, metric_value))| (key.key().name().to_owned(), metric_value))
        .collect();
    (value, metrics_by_name)
}

/// Builds an index used only to drive the boundary scorer.
fn index() -> CpuHnsw {
    CpuHnsw::with_capacity(HnswParams::new(16, 200).expect("params"), TRIM_POINTS).expect("index")
}

/// Returns the number of recorded duration samples, when there are any.
fn duration_samples(metrics_by_name: &HashMap<String, DebugValue>) -> Option<usize> {
    match metrics_by_name.get("chutoro.hnsw.trim.duration_seconds") {
        Some(DebugValue::Histogram(samples)) => Some(samples.len()),
        _ => None,
    }
}

#[test]
fn an_empty_trim_batch_records_nothing() {
    let source = DenseSource::synthetic(TRIM_POINTS);
    let index = index();
    let (scored, metrics_by_name) =
        record_trim_metrics(|| index.score_trim_jobs_with_metrics(Vec::new(), &source));

    let scored_jobs = scored.expect("an empty batch must score cleanly");
    assert!(
        scored_jobs.is_empty(),
        "an empty batch must produce no results"
    );
    assert!(
        !metrics_by_name.contains_key("chutoro.hnsw.trim.jobs_submitted_total"),
        "an empty batch must not count a job it was never given"
    );
    assert_eq!(
        duration_samples(&metrics_by_name),
        None,
        "an empty batch must not record a duration sample"
    );
}

#[test]
fn a_successful_trim_batch_records_every_submitted_job() {
    let source = DenseSource::synthetic(TRIM_POINTS);
    let index = index();
    let jobs: Vec<TrimJob> = (0..TRIM_JOBS).map(DenseSource::trim_job).collect();
    let (scored, metrics_by_name) =
        record_trim_metrics(|| index.score_trim_jobs_with_metrics(jobs, &source));

    let scored_jobs = scored.expect("trim scoring must succeed");
    assert_eq!(
        scored_jobs.len(),
        TRIM_JOBS,
        "every trim job must be scored"
    );
    assert_eq!(
        metrics_by_name.get("chutoro.hnsw.trim.jobs_submitted_total"),
        Some(&DebugValue::Counter(TRIM_JOBS as u64)),
        "one scored batch must count every job it was given"
    );
    assert_eq!(
        duration_samples(&metrics_by_name),
        Some(1),
        "one scored batch must record one duration sample"
    );
}

#[test]
fn a_failed_trim_batch_records_every_submitted_job() {
    let source = DenseSource::synthetic(TRIM_POINTS);
    let index = index();
    let mut jobs: Vec<TrimJob> = (0..TRIM_JOBS).map(DenseSource::trim_job).collect();
    // Scoring stops at the first invalid job, so the counter can only equal
    // the submitted batch size if the boundary records before propagating.
    jobs.push(TrimJob {
        node: TRIM_JOBS,
        ctx: EdgeContext {
            level: 0,
            max_connections: 8,
        },
        candidates: vec![0, 1],
        sequences: vec![0],
    });
    let submitted = jobs.len();
    let (scored, metrics_by_name) =
        record_trim_metrics(|| index.score_trim_jobs_with_metrics(jobs, &source));

    assert!(
        scored.is_err(),
        "mismatched candidate and sequence counts must fail scoring"
    );
    assert_eq!(
        metrics_by_name.get("chutoro.hnsw.trim.jobs_submitted_total"),
        Some(&DebugValue::Counter(submitted as u64)),
        "a failed batch must still count every job it was given"
    );
    assert_eq!(
        duration_samples(&metrics_by_name),
        Some(1),
        "a failed batch must still record its duration"
    );
}
