//! Filesystem tests for the capability-scoped staging lock.
//!
//! Covers acquisition, release, reacquisition, invalid names, and a hostile
//! non-regular lockfile path. Process-contention behaviour lives beside this
//! file in `staging_lock_contention.rs`.

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

use chutoro_bench_datasets::{RecipeError, StagingLock};
use rstest::{fixture, rstest};

mod support;

use support::{StagingRoot, hold_externally, make_fifo, policy};

#[fixture]
fn root_res() -> io::Result<StagingRoot> {
    StagingRoot::new()
}

#[rstest]
fn acquire_and_release_allows_reacquisition(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    let lock = root.lock();
    let first = lock
        .acquire(
            "dataset.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect("uncontended acquisition should succeed");
    assert!(first.path().ends_with("dataset.lock"));
    drop(first);

    let second = lock
        .acquire(
            "dataset.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect("reacquisition after release should succeed");
    assert_eq!(second.path(), lock.root().join("dataset.lock"));
}

#[rstest]
fn lockfile_persists_after_release(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    let lock = root.lock();
    drop(
        lock.acquire(
            "persistent.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect("acquisition should succeed"),
    );

    assert!(
        root.contains("persistent.lock"),
        "the lockfile must survive release so its pathname stays stable"
    );
}

#[rstest]
fn second_in_process_acquisition_is_rejected_immediately(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    let lock = root.lock();
    let _held = lock
        .acquire(
            "duplicate.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect("first acquisition should succeed");

    let error = lock
        .acquire(
            "duplicate.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect_err("a second in-process acquisition should fail");

    assert!(matches!(
        error,
        RecipeError::StagingLockContended {
            waited_millis: 0,
            ..
        }
    ));
}

#[rstest]
fn rejected_duplicate_leaves_the_first_lock_usable(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    let lock = root.lock();
    let held = lock
        .acquire(
            "shared.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect("first acquisition should succeed");
    let duplicate = lock
        .acquire(
            "shared.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect_err("a second in-process acquisition should fail");
    assert!(
        matches!(
            duplicate,
            RecipeError::StagingLockContended {
                waited_millis: 0,
                ..
            }
        ),
        "the duplicate must be rejected before any wait"
    );
    assert!(
        hold_externally(&root.join("shared.lock")).is_err(),
        "the first guard must still hold the lock against an independent descriptor"
    );

    drop(held);
    let reacquired = lock
        .acquire(
            "shared.lock",
            policy(Duration::from_millis(50)).expect("policy"),
        )
        .expect("the failed duplicate must not leave a registry entry behind");
    drop(reacquired);
}

#[rstest]
#[case::parent_directory("../escape.lock")]
#[case::nested_path("nested/dataset.lock")]
#[case::missing_suffix("dataset.txt")]
#[case::empty("")]
#[case::absolute("/tmp/dataset.lock")]
#[case::backslash("dataset\\lock.lock")]
#[case::dot(".")]
fn invalid_names_are_rejected(root_res: io::Result<StagingRoot>, #[case] name: &str) {
    let root = root_res.expect("temporary staging root should be created");
    let error = root
        .lock()
        .acquire(name, policy(Duration::from_millis(20)).expect("policy"))
        .expect_err("invalid lock name should be rejected");

    assert!(matches!(error, RecipeError::StagingLockInvalidName(_)));
}

#[rstest]
fn missing_root_reports_an_io_failure(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    let absent = root.join("absent");
    let error = StagingLock::new(absent)
        .acquire(
            "dataset.lock",
            policy(Duration::from_millis(20)).expect("policy"),
        )
        .expect_err("a missing capability root should fail");

    assert!(matches!(error, RecipeError::StagingLockIo { .. }));
}

#[rstest]
fn fifo_lockfile_is_rejected_without_blocking(root_res: io::Result<StagingRoot>) {
    let root = root_res.expect("temporary staging root should be created");
    make_fifo(&root.join("fifo.lock")).expect("the FIFO fixture should be created");

    // Run the acquisition on a worker so a blocking open fails this test at the
    // watchdog instead of leaving the suite to its 60-second slow timeout.
    let lock = root.lock();
    let policy = policy(Duration::from_millis(20)).expect("policy");
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let outcome = lock.acquire("fifo.lock", policy).map(|_guard| ());
        drop(sender.send(outcome));
    });

    match receiver.recv_timeout(Duration::from_secs(2)) {
        Ok(Err(RecipeError::StagingLockIo { .. })) => {}
        Ok(Err(other)) => panic!("unexpected error: {other}"),
        Ok(Ok(())) => panic!("a FIFO lockfile must not be acquired"),
        Err(error) => panic!("the FIFO open blocked past the watchdog: {error}"),
    }
    worker.join().expect("watchdog worker should finish");
}
