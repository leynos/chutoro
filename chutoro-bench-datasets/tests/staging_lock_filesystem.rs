//! Filesystem tests for the capability-scoped staging lock.

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

use camino::{Utf8Path, Utf8PathBuf};
use cap_std::{ambient_authority, fs_utf8::Dir};
use chutoro_bench_datasets::{LockPolicy, LockPolicyError, RecipeError, StagingLock};
use rstest::{fixture, rstest};
use tempfile::TempDir;
use tracing_test::traced_test;

/// Poll interval used by every test, short enough to keep the suite quick.
const TEST_POLL: Duration = Duration::from_millis(2);

/// Temporary capability root that removes itself when the test ends.
struct StagingRoot {
    /// Owner of the temporary directory.
    _temp: TempDir,
    /// UTF-8 path of the temporary directory.
    path: Utf8PathBuf,
}

impl StagingRoot {
    /// Create a fresh staging root.
    fn new() -> io::Result<Self> {
        let temp = tempfile::tempdir()?;
        let path = Utf8PathBuf::from_path_buf(temp.path().to_path_buf()).map_err(|path| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("temporary path is not UTF-8: {}", path.display()),
            )
        })?;
        Ok(Self { _temp: temp, path })
    }

    /// Build a lock factory rooted at this directory.
    fn lock(&self) -> StagingLock {
        StagingLock::new(self.path.clone())
    }

    /// Return the absolute path of one file inside the root.
    fn join(&self, name: &str) -> Utf8PathBuf {
        self.path.join(name)
    }

    /// Return true when `name` exists inside the root.
    fn contains(&self, name: &str) -> bool {
        Dir::open_ambient_dir(&self.path, ambient_authority())
            .and_then(|dir| dir.metadata(name))
            .is_ok()
    }
}

/// Build a policy with the shared test poll interval.
fn policy(timeout: Duration) -> Result<LockPolicy, LockPolicyError> {
    LockPolicy::new(timeout).and_then(|policy| policy.with_poll_interval(TEST_POLL))
}

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

/// Hold `name` with an independent descriptor, bypassing the registry.
fn hold_externally(path: &Utf8Path) -> io::Result<cap_std::fs_utf8::File> {
    use cap_std::io_lifetimes::AsFd;

    let dir = Dir::open_ambient_dir(
        path.parent().unwrap_or_else(|| Utf8Path::new(".")),
        ambient_authority(),
    )?;
    let mut options = cap_std::fs_utf8::OpenOptions::new();
    options.write(true).create(true);
    let file = dir.open_with(path.file_name().unwrap_or("dataset.lock"), &options)?;
    rustix::fs::flock(
        cap_std::fs_utf8::File::as_fd(&file),
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    )
    .map_err(io::Error::from)?;
    Ok(file)
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

/// Create a FIFO at `path` through `mkfifo(1)`, which avoids a libc binding.
fn make_fifo(path: &Utf8Path) -> io::Result<()> {
    let status = std::process::Command::new("mkfifo")
        .arg(path.as_str())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("mkfifo failed with {status}")))
    }
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
