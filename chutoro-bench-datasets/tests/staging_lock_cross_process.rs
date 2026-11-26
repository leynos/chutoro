//! Cross-process exclusion test for the capability-scoped staging lock.
//!
//! The parent test re-runs this test binary with an exact filter. The child
//! acquires the lock, signals readiness, waits for a release marker, and then
//! exits. The parent proves that a live external holder produces a bounded
//! timeout and that the lock is acquirable again once the child has exited.

#![cfg(all(feature = "http", target_os = "linux"))]

use std::{
    io,
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

use camino::{Utf8Path, Utf8PathBuf};
use cap_std::{ambient_authority, fs_utf8::Dir};
use chutoro_bench_datasets::{LockPolicy, RecipeError, StagingLock};
use mockable::{DefaultEnv, Env};
use tempfile::TempDir;

/// Environment variable that marks the child as the lock holder.
const HOLDER_ENV: &str = "CHUTORO_STAGING_LOCK_HOLDER_ROOT";
/// Exact libtest filter for the holder role.
///
/// Integration-test names carry no module prefix, so the filter is the bare
/// function name and `--exact` selects it alone.
const HOLDER_TEST: &str = "holder_holds_lock_until_released";
/// Lockfile shared by parent and child.
const LOCK_NAME: &str = "cross-process.lock";
/// Marker the child writes after acquiring the lock.
const READY_MARKER: &str = "holder.ready";
/// Marker the parent writes to release the child.
const RELEASE_MARKER: &str = "holder.release";
/// Poll interval for marker files and child exit.
const WAIT_STEP: Duration = Duration::from_millis(10);
/// Upper bound on any cross-process wait.
const WAIT_TIMEOUT: Duration = Duration::from_secs(15);
/// Timeout used to prove contention while the child holds the lock.
const CONTENTION_TIMEOUT: Duration = Duration::from_millis(60);

/// Child-role test: hold the lock until the parent writes the release marker.
#[test]
fn holder_holds_lock_until_released() {
    let Some(root) = holder_root() else {
        return;
    };
    let policy = LockPolicy::new(Duration::from_secs(5))
        .and_then(|policy| policy.with_poll_interval(Duration::from_millis(2)))
        .expect("holder policy should be valid");
    let guard = StagingLock::new(root.as_path())
        .acquire(LOCK_NAME, policy)
        .expect("the holder child should acquire the lock");
    let dir = Dir::open_ambient_dir(&root, ambient_authority())
        .expect("the holder child should open the fixture root");
    dir.write(READY_MARKER, b"ready")
        .expect("the holder child should write the ready marker");

    wait_for_marker(&dir, RELEASE_MARKER, WAIT_TIMEOUT)
        .expect("the parent should release the holder child");
    drop(guard);
}

/// Parent-role test: time out while the child holds the lock, then reacquire.
#[test]
fn parent_times_out_while_child_holds_then_reacquires() {
    let (_temp, root) = capability_root().expect("temporary root should be created");
    let mut command = Command::new(std::env::current_exe().expect("test binary should resolve"));
    command
        .args(["--exact", HOLDER_TEST, "--nocapture"])
        .env(HOLDER_ENV, root.as_str())
        .stdout(Stdio::null());
    let spawned = command.spawn().expect("holder child should start");
    let mut child = ChildGuard::new(spawned);

    let dir = open_root(&root).expect("fixture root should open");
    child
        .wait_for_marker(&dir, READY_MARKER, WAIT_TIMEOUT)
        .expect("the holder child should become ready");

    let lock = StagingLock::new(root.as_path());
    let policy = LockPolicy::new(CONTENTION_TIMEOUT)
        .and_then(|policy| policy.with_poll_interval(Duration::from_millis(2)))
        .expect("contention policy should be valid");
    let error = match lock.acquire(LOCK_NAME, policy) {
        Err(error) => error,
        Ok(_guard) => panic!("a lock held by the child should time out"),
    };
    assert!(
        matches!(
            error,
            RecipeError::StagingLockContended {
                waited_millis,
                ..
            } if u128::from(waited_millis) >= CONTENTION_TIMEOUT.as_millis()
        ),
        "unexpected error: {error}"
    );

    dir.write(RELEASE_MARKER, b"release")
        .expect("the parent should write the release marker");
    let status = child
        .wait_bounded(WAIT_TIMEOUT)
        .expect("the holder child should exit");
    assert!(status.success(), "the holder child failed: {status}");

    let reacquired = lock
        .acquire(LOCK_NAME, policy)
        .expect("the parent should reacquire once the child has exited");
    drop(reacquired);
}

/// Read the holder-role marker through the injectable environment reader.
fn holder_root() -> Option<Utf8PathBuf> {
    DefaultEnv.string(HOLDER_ENV).map(Utf8PathBuf::from)
}

/// Create a temporary capability root.
fn capability_root() -> io::Result<(TempDir, Utf8PathBuf)> {
    let temp = tempfile::tempdir()?;
    let path = Utf8PathBuf::from_path_buf(temp.path().to_path_buf()).map_err(|path| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("temporary root is not UTF-8: {}", path.display()),
        )
    })?;
    Ok((temp, path))
}

/// Open a capability root for marker reads and writes.
fn open_root(path: &Utf8Path) -> io::Result<Dir> {
    Dir::open_ambient_dir(path, ambient_authority())
}

/// Poll for a marker file until it appears or the timeout expires.
fn wait_for_marker(dir: &Dir, name: &str, timeout: Duration) -> io::Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        match dir.metadata(name) {
            Ok(_) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("marker {name} did not appear"),
            ));
        }
        thread::sleep(WAIT_STEP);
    }
}

/// Child process that is killed and reaped unless it exits cleanly.
struct ChildGuard {
    /// The live child, or `None` once it has been reaped.
    child: Option<Child>,
}

impl ChildGuard {
    /// Take ownership of a spawned child.
    #[must_use]
    const fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    /// Poll for `name` while the child runs, failing fast if it exits first.
    fn wait_for_marker(&mut self, dir: &Dir, name: &str, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            match dir.metadata(name) {
                Ok(_) => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            if let Some(status) = self.try_wait()? {
                return Err(io::Error::other(format!(
                    "holder child exited before writing {name}: {status}"
                )));
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("marker {name} did not appear"),
                ));
            }
            thread::sleep(WAIT_STEP);
        }
    }

    /// Reap the child if it has already exited.
    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        let Some(child) = self.child.as_mut() else {
            return Ok(None);
        };
        let status = child.try_wait()?;
        if status.is_some() {
            self.child = None;
        }
        Ok(status)
    }

    /// Wait for the child to exit, bounded by `timeout`.
    fn wait_bounded(&mut self, timeout: Duration) -> io::Result<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            let status = self
                .child
                .as_mut()
                .ok_or_else(|| io::Error::other("child already reaped"))?
                .try_wait()?;
            if let Some(exit_status) = status {
                self.child = None;
                return Ok(exit_status);
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "holder child did not exit",
                ));
            }
            thread::sleep(WAIT_STEP);
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            drop(child.kill());
            drop(child.wait());
        }
    }
}
