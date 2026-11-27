//! Regression guard for the callsite-interest pin used by tracing-capture
//! tests.
//!
//! `tracing-core` caches a callsite's interest the first time the callsite is
//! recorded. Without the pin, an unscoped thread that touches `core.run` while
//! the capture thread's recording subscriber is the only live dispatch caches
//! `Interest::never` for the whole process, and the recording subscriber then
//! misses the span. The regression depends entirely on which thread first
//! touches the callsite, so this module holds a single test: a second test in
//! the same binary would observe whatever interest the first one cached. The
//! same one-test rule, for the same process-global reason, is explained in
//! `chutoro-cli/tests/logging_conflict.rs`.
//!
//! The worker starts only after the capture thread has installed its recording
//! subscriber, so this test covers the ordering the pin guarantees. It does not
//! cover a registration that straddles dispatch setup, which is a
//! `tracing-core` race described on `CallsiteInterestPin` and cannot be closed
//! from a downstream test.

mod common;

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use chutoro_core::{Chutoro, ChutoroBuilder, ChutoroError};
use chutoro_test_support::tracing::{RecordingLayer, pin_callsite_interest};
use common::Dummy;
use tracing::Level;
use tracing_subscriber::layer::SubscriberExt;

/// Builds the default orchestrator that the empty-source scenario exercises.
///
/// The helper returns the builder outcome so the `#[test]` bodies own the
/// decision to fail, as the fixture policy requires.
fn empty_source_chutoro() -> Result<Chutoro, ChutoroError> {
    ChutoroBuilder::new().build()
}

/// Upper bound on each worker hand-off, so a stalled thread fails the test
/// instead of hanging the shared-process run indefinitely.
const HAND_OFF_TIMEOUT: Duration = Duration::from_secs(30);

/// Verifies that an unscoped first touch of `core.run` does not hide the span
/// from a recording subscriber installed on another thread.
#[test]
fn pinned_interest_keeps_the_core_run_span_visible() {
    let _pin = pin_callsite_interest();
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());
    let (start_tx, start_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();

    let (capture_result, unscoped_result) = tracing::subscriber::with_default(subscriber, || {
        let unscoped = thread::spawn(move || {
            start_rx
                .recv_timeout(HAND_OFF_TIMEOUT)
                .expect("the capture thread must release the unscoped worker");
            let result = empty_source_chutoro()
                .expect("default configuration must be valid")
                .run(&Dummy::new(vec![]));
            done_tx
                .send(())
                .expect("the capture thread must await the unscoped first touch");
            result
        });

        start_tx
            .send(())
            .expect("the unscoped worker must still be waiting to start");
        done_rx
            .recv_timeout(HAND_OFF_TIMEOUT)
            .expect("the unscoped worker must report its first touch");
        let capture_result = empty_source_chutoro()
            .expect("default configuration must be valid")
            .run(&Dummy::new(vec![]));
        let unscoped_result = unscoped.join().expect("the unscoped worker must not panic");
        (capture_result, unscoped_result)
    });

    let capture_error =
        capture_result.expect_err("the empty source must be rejected on the capture thread");
    assert!(matches!(capture_error, ChutoroError::EmptySource { .. }));
    let unscoped_error = unscoped_result.expect_err("the empty source must be rejected unscoped");
    assert!(matches!(unscoped_error, ChutoroError::EmptySource { .. }));

    let spans = layer.spans();
    let events = layer.events();
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
                && event
                    .fields
                    .get("error_code")
                    .is_some_and(|value| value == "CHUTORO_EMPTY_SOURCE")
        }),
        "the empty-source warning must be recorded; spans: {spans:?}, events: {events:?}"
    );
}
