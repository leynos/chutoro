//! Metrics tests for the CPU pipeline stage timings.

use metrics_util::debugging::{DebugValue, DebuggingRecorder};

use crate::HnswParams;

use super::super::run_cpu_pipeline_with_len;
use super::fixture::{ClusteredSource, ITEM_COUNT, MIN_CLUSTER_SIZE};

/// Histogram the pipeline must sample once per stage.
const STAGE_SECONDS: &str = "chutoro.cpu_pipeline.stage_seconds";

#[test]
fn every_stage_records_a_latency_sample() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let source = ClusteredSource::new();
    let params = HnswParams::new(4, 16).expect("fixture parameters must be valid");

    ::metrics::with_local_recorder(&recorder, || {
        run_cpu_pipeline_with_len(&source, ITEM_COUNT, MIN_CLUSTER_SIZE, &params)
            .expect("production pipeline must succeed");
    });

    let mut stages: Vec<(String, usize)> = snapshotter
        .snapshot()
        .into_hashmap()
        .into_iter()
        .filter(|(key, _)| key.key().name() == STAGE_SECONDS)
        .map(|(key, (_, _, value))| {
            let labels: Vec<String> = key
                .key()
                .labels()
                .map(|label| label.value().to_owned())
                .collect();
            let count = match value {
                DebugValue::Histogram(samples) => samples.len(),
                other => panic!("stage metric must be a histogram, got {other:?}"),
            };
            (labels.join(","), count)
        })
        .collect();
    stages.sort();

    assert_eq!(
        stages,
        [
            ("kruskal".to_owned(), 1),
            ("mutual_reachability".to_owned(), 1)
        ],
        "each pipeline stage must record exactly one bounded latency histogram sample"
    );
}
