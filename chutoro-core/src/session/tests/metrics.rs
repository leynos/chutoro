//! Metrics-specific session tests.

use rstest::rstest;

use super::common::{make_session, session_builder};
use crate::ChutoroBuilder;

#[rstest]
fn append_records_deterministic_latency_via_clock_seam(session_builder: ChutoroBuilder) {
    use std::{sync::Arc, time::Duration};

    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    // Install a local debugging recorder for this test only.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        // Build a session with a fixed clock that reports exactly 5 ms per point.
        let (session, _) = make_session(session_builder, 4);
        let clock = Arc::new(crate::session::clock::FixedMonotonicClock::with_elapsed(
            Duration::from_millis(5),
        ));
        let mut session = session.with_clock_for_test(clock);

        session.append(&[0]).expect("single append must succeed");
    });

    let snapshot = snapshotter.snapshot();
    let histogram = snapshot
        .into_hashmap()
        .into_iter()
        .find(|(key, _)| key.key().name() == "chutoro.session.append.point_seconds")
        .map(|(_, (_, _, value))| value)
        .expect("histogram must be recorded");

    if let DebugValue::Histogram(buckets) = histogram {
        let recorded = buckets
            .first()
            .expect("at least one sample must be recorded")
            .into_inner();
        assert!(
            (recorded - 0.005_f64).abs() < 1e-6,
            "recorded latency must match the fixed clock: expected 0.005 s, got {recorded}"
        );
    } else {
        panic!("expected a Histogram metric value, got {histogram:?}");
    }
}
