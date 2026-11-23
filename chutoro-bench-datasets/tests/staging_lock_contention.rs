//! Contention tests for the capability-scoped staging lock.
//!
//! Each case holds an independent `flock` on the lockfile so the acquisition
//! must genuinely wait, then checks the bounded timeout, the single warning,
//! and reacquisition after release.

#![cfg(all(
    feature = "http",
    feature = "testing",
    not(any(
        windows,
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]

use std::{io, time::Duration};

use chutoro_bench_datasets::RecipeError;
use rstest::{fixture, rstest};
use tracing_test::traced_test;

mod support;

use support::{StagingRoot, hold_externally, policy};

#[fixture]
fn root_res() -> io::Result<StagingRoot> {
    StagingRoot::new()
}

#[traced_test]
#[rstest]
fn externally_held_lock_times_out_then_allows_reacquisition(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    let lock = root.lock();
    let blocking = root.join("blocked.lock");
    let external = hold_externally(&blocking).expect("the external lock should be acquired");

    let error = lock
        .acquire(
            "blocked.lock",
            policy(Duration::from_millis(40)).expect("policy"),
        )
        .expect_err("a lock held by another descriptor should time out");
    assert!(
        matches!(
            error,
            RecipeError::StagingLockContended {
                waited_millis,
                ..
            } if waited_millis >= 40
        ),
        "unexpected error: {error}"
    );
    assert!(logs_contain("staging lock is held"));

    drop(external);
    let reacquired = lock
        .acquire(
            "blocked.lock",
            policy(Duration::from_millis(200)).expect("policy"),
        )
        .expect("the lock should be acquirable once the external holder releases it");
    drop(reacquired);
}

#[traced_test]
#[rstest]
fn contention_warns_only_once(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    let lock = root.lock();
    let blocking = root.join("warned.lock");
    let external = hold_externally(&blocking).expect("the external lock should be acquired");

    let _ = lock
        .acquire(
            "warned.lock",
            policy(Duration::from_millis(40)).expect("policy"),
        )
        .expect_err("the held lock should time out");
    drop(external);

    logs_assert(|lines: &[&str]| {
        let warnings = lines
            .iter()
            .filter(|line| line.contains("staging lock is held"))
            .count();
        if warnings == 1 {
            Ok(())
        } else {
            Err(format!(
                "expected exactly one contention warning, saw {warnings}"
            ))
        }
    });
}

#[traced_test]
#[rstest]
fn deadline_edge_contention_still_warns(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    let lock = root.lock();
    let blocking = root.join("deadline-edge.lock");
    let external = hold_externally(&blocking).expect("the external lock should be acquired");

    let error = lock
        .acquire(
            "deadline-edge.lock",
            policy(Duration::from_nanos(1)).expect("policy"),
        )
        .expect_err("the first contended attempt is already past the deadline");

    assert!(
        matches!(error, RecipeError::StagingLockContended { .. }),
        "unexpected error: {error}"
    );
    assert!(
        logs_contain("staging lock is held"),
        "contention observed only at the deadline must still warn"
    );
    drop(external);
}

#[rstest]
fn concurrent_threads_serialize_on_the_registry(root_res: io::Result<StagingRoot>) {
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    let root = root_res.expect("temporary staging root should be created");
    let shared = Arc::new(root.lock());
    let barrier = Arc::new(Barrier::new(4));
    let successes = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for _ in 0..4 {
        let lock = Arc::clone(&shared);
        let worker_barrier = Arc::clone(&barrier);
        let worker_successes = Arc::clone(&successes);
        handles.push(std::thread::spawn(move || {
            let guard = lock
                .acquire(
                    "threaded.lock",
                    policy(Duration::from_millis(50)).expect("policy"),
                )
                .ok();
            if guard.is_some() {
                worker_successes.fetch_add(1, Ordering::SeqCst);
            }
            // Hold a winning guard until every worker has attempted, so the
            // registry, not scheduling, decides the outcome.
            worker_barrier.wait();
            drop(guard);
        }));
    }
    for handle in handles {
        handle.join().expect("worker thread should finish");
    }

    assert_eq!(
        successes.load(Ordering::SeqCst),
        1,
        "exactly one of four racing threads should hold the lock"
    );
    let reacquired = shared
        .acquire(
            "threaded.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect("the registry must be clean once every worker has dropped its guard");
    drop(reacquired);
}
