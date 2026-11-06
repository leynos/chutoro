//! Metrics tests for the CPU pipeline stage timings.

use std::num::NonZeroUsize;

use metrics_util::debugging::DebuggingRecorder;

use crate::HnswParams;

use super::fixture::{ClusteredSource, ITEM_COUNT};
use super::super::run_cpu_pipeline_with_len;

/// Histogram the pipeline must sample once per stage.
const STAGE_SECONDS: &str = "chutoro.cpu_pipeline.stage_seconds";

#[test]
fn every_stage_records_a_latency_sample() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let source = ClusteredSource::new();
    let min_cluster_size = NonZeroUsize::new(2).expect("literal 2 is non-zero");
    let params = HnswParams::new(4, 16).expect("fixture parameters must be valid");

    ::metrics::with_local_recorder(&recorder, || {
        run_cpu_pipeline_with_len(&source, ITEM_COUNT, min_cluster_size, &params)
            .expect("production pipeline must succeed");
    });

    let mut stages: Vec<String> = snapshotter
        .snapshot()
        .into_hashmap()
        .into_iter()
        .filter(|(key, _)| key.key().name() == STAGE_SECONDS)
        .map(|(key, _)| {
            let labels: Vec<String> = key
                .key()
                .labels()
                .map(|label| label.value().to_owned())
                .collect();
            labels.join(",")
        })
        .collect();
    stages.sort();

    assert_eq!(
        stages,
        ["kruskal", "mutual_reachability"],
        "each pipeline stage must record one bounded latency histogram sample"
    );
}