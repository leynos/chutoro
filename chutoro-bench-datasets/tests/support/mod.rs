//! Shared fixtures for the staging-lock filesystem tests.
//!
//! This module lives in a `tests/support` subdirectory so Cargo does not treat
//! it as a standalone test crate; consumers pull it in with `mod support;`.

#![allow(dead_code, reason = "each test binary uses a different subset")]

use std::{io, time::Duration};

use camino::{Utf8Path, Utf8PathBuf};
use cap_std::{ambient_authority, fs_utf8::Dir};
use chutoro_bench_datasets::{LockPolicy, LockPolicyError, StagingLock};
use tempfile::TempDir;

/// Poll interval used by every test, short enough to keep the suite quick.
pub const TEST_POLL: Duration = Duration::from_millis(2);

/// Temporary capability root that removes itself when the test ends.
pub struct StagingRoot {
    /// Owner of the temporary directory.
    _temp: TempDir,
    /// UTF-8 path of the temporary directory.
    path: Utf8PathBuf,
}

impl StagingRoot {
    /// Create a fresh staging root.
    pub fn new() -> io::Result<Self> {
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
    pub fn lock(&self) -> StagingLock {
        StagingLock::new(self.path.clone())
    }

    /// Return the absolute path of one file inside the root.
    pub fn join(&self, name: &str) -> Utf8PathBuf {
        self.path.join(name)
    }

    /// Return true when `name` exists inside the root.
    pub fn contains(&self, name: &str) -> bool {
        Dir::open_ambient_dir(&self.path, ambient_authority())
            .and_then(|dir| dir.metadata(name))
            .is_ok()
    }
}

/// Build a policy with the shared test poll interval.
pub fn policy(timeout: Duration) -> Result<LockPolicy, LockPolicyError> {
    LockPolicy::new(timeout).and_then(|policy| policy.with_poll_interval(TEST_POLL))
}

/// Hold `name` with an independent descriptor, bypassing the registry.
pub fn hold_externally(path: &Utf8Path) -> io::Result<cap_std::fs_utf8::File> {
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

/// Create a FIFO at `path` through `mkfifo(1)`, which avoids a libc binding.
pub fn make_fifo(path: &Utf8Path) -> io::Result<()> {
    let status = std::process::Command::new("mkfifo")
        .arg(path.as_str())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("mkfifo failed with {status}")))
    }
}
