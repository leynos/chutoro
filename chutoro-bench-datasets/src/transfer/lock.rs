//! Capability-scoped adapter for advisory staging-file locks.
//!
//! The adapter owns the filesystem and the clock; [`LockPolicy`] owns the
//! retry decision. A guard owns the only descriptor for the lockfile, its
//! registry key, and the registry entry itself.

use std::{
    collections::HashSet,
    fmt, io,
    sync::{Mutex, MutexGuard, OnceLock, PoisonError},
    thread,
    time::{Duration, Instant},
};

use camino::{Utf8Component, Utf8Path, Utf8PathBuf};
use cap_std::{
    ambient_authority,
    fs_utf8::{Dir, File, OpenOptions},
};
use tracing::warn;

use super::platform::{self, FLOCK_SUPPORTED};
use crate::{RecipeError, transfer::LockPolicy, transfer::LockStep};

/// Suffix every staging lockfile name must carry.
const LOCK_SUFFIX: &str = ".lock";

/// Capability-rooted factory for advisory staging locks.
///
/// The root is the directory that contains the staging files and their stable
/// `.lock` companions. It must already exist and must live on a local
/// filesystem; see [`crate::transfer`] for the `flock` caveats.
///
/// # Examples
///
/// ```
/// use chutoro_bench_datasets::StagingLock;
///
/// let lock = StagingLock::new("target/staging");
/// assert_eq!(lock.root().as_str(), "target/staging");
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagingLock {
    /// Directory that contains the staging files and their lockfiles.
    root: Utf8PathBuf,
}

impl StagingLock {
    /// Create a lock factory rooted at `root`.
    #[must_use]
    pub fn new(root: impl Into<Utf8PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Borrow the capability root.
    #[must_use]
    pub fn root(&self) -> &Utf8Path {
        &self.root
    }

    /// Acquire the named staging lock under the capability root.
    ///
    /// The name must be one relative UTF-8 component ending in `.lock`. The
    /// file is created if it does not exist, opened for writing without
    /// truncation, and locked with a non-blocking exclusive `flock`. The call
    /// retries until the lock is granted or [`LockPolicy`] reports a timeout.
    ///
    /// Keep the returned guard alive for as long as the staging operation
    /// needs exclusion; dropping it releases the lock.
    ///
    /// # Errors
    ///
    /// - [`RecipeError::StagingLockInvalidName`] when `name` is not one
    ///   relative UTF-8 `.lock` component.
    /// - [`RecipeError::StagingLockContended`] when this process already holds
    ///   the lock, or when another holder outlasts the policy timeout. The
    ///   `waited_millis` field is zero for the in-process case.
    /// - [`RecipeError::StagingLockIo`] when opening, creating, or locking the
    ///   file fails for another reason.
    /// - [`RecipeError::StagingLockUnsupportedPlatform`] on a target without
    ///   `flock`; no filesystem operation is attempted.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::time::Duration;
    /// use chutoro_bench_datasets::{LockPolicy, RecipeError, StagingLock};
    ///
    /// let lock = StagingLock::new("target/staging");
    /// let policy =
    ///     LockPolicy::new(Duration::from_secs(5)).map_err(|error| RecipeError::other(error))?;
    /// let guard = lock.acquire("dataset.lock", policy)?;
    /// assert!(guard.path().ends_with("dataset.lock"));
    /// # Ok::<(), RecipeError>(())
    /// ```
    pub fn acquire(&self, name: &str, policy: LockPolicy) -> Result<StagingLockGuard, RecipeError> {
        let file_name = validate_lock_name(name)?;
        if !FLOCK_SUPPORTED {
            return Err(RecipeError::staging_lock_unsupported_platform());
        }
        let path = self.root.join(file_name);
        let registration = Registration::acquire(path.clone())?;
        let dir = Dir::open_ambient_dir(&self.root, ambient_authority())
            .map_err(|error| RecipeError::staging_lock_io(path.as_str(), error))?;
        let file = open_lockfile(&dir, file_name)
            .map_err(|error| RecipeError::staging_lock_io(path.as_str(), error))?;
        acquire_file(file, &path, policy, registration)
    }
}

/// RAII guard for one held staging lock.
///
/// Dropping the guard unlocks the advisory lock, removes the process-wide
/// registry entry, and closes the descriptor. The lockfile is left in place.
#[must_use = "dropping the guard releases the staging lock"]
pub struct StagingLockGuard {
    /// The only descriptor for the lockfile held by this crate.
    file: File,
    /// Registry key and diagnostic path for the lockfile.
    path: Utf8PathBuf,
}

impl StagingLockGuard {
    /// Wrap a locked file and its registry key.
    const fn new(file: File, path: Utf8PathBuf) -> Self {
        Self { file, path }
    }

    /// Return the lockfile path this guard holds.
    #[must_use]
    pub fn path(&self) -> &Utf8Path {
        &self.path
    }
}

impl fmt::Debug for StagingLockGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StagingLockGuard")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl Drop for StagingLockGuard {
    fn drop(&mut self) {
        platform::unlock(&self.file);
        release_path(&self.path);
        // The descriptor closes after this method returns, which also releases
        // any lock the kernel still associates with it.
    }
}

/// Process-wide registry of staging lockfiles held by this process.
///
/// `flock` is per open file description, so it cannot distinguish two
/// acquisitions in one process reliably. The registry turns that case into an
/// immediate, deterministic failure before the first system call.
fn registry() -> &'static Mutex<HashSet<Utf8PathBuf>> {
    static REGISTRY: OnceLock<Mutex<HashSet<Utf8PathBuf>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Lock the process-wide registry, recovering a poisoned mutex.
fn lock_registry() -> MutexGuard<'static, HashSet<Utf8PathBuf>> {
    registry().lock().unwrap_or_else(PoisonError::into_inner)
}

/// Remove one path from the process-wide registry.
fn release_path(path: &Utf8Path) {
    let mut entries = lock_registry();
    entries.remove(path);
}

/// Registry entry that removes itself unless ownership moves to a guard.
struct Registration {
    /// Registry key for the lockfile.
    path: Utf8PathBuf,
    /// Whether the entry still needs to be removed on drop.
    active: bool,
}

impl Registration {
    /// Insert the path, rejecting a duplicate immediately and visibly.
    ///
    /// This rejection bypasses the bounded `flock` loop, so it cannot use the
    /// policy's single-warning slot. It is still contention, and an
    /// unexplained immediate failure sends a caller looking in the wrong
    /// place, so it gets its own warning on every rejected acquisition.
    fn acquire(path: Utf8PathBuf) -> Result<Self, RecipeError> {
        // The guard must be released before `warn!`: an enabled tracing
        // subscriber runs inside the event call and may acquire another
        // staging lock, which would re-enter this mutex and deadlock.
        let inserted = {
            let mut entries = lock_registry();
            entries.insert(path.clone())
        };
        if !inserted {
            warn!(
                path = path.as_str(),
                "staging lock is already held by this process"
            );
            return Err(RecipeError::staging_lock_contended(path.as_str(), 0));
        }
        Ok(Self { path, active: true })
    }

    /// Mark the entry as owned by the returned guard.
    fn into_path(mut self) -> Utf8PathBuf {
        self.active = false;
        std::mem::take(&mut self.path)
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if self.active {
            release_path(&self.path);
        }
    }
}

/// Validate and return a single relative UTF-8 `.lock` component.
fn validate_lock_name(name: &str) -> Result<&Utf8Path, RecipeError> {
    let path = Utf8Path::new(name);
    if is_valid_lock_name(name, path) {
        Ok(path)
    } else {
        Err(RecipeError::staging_lock_invalid_name(name))
    }
}

/// Whether `name` is one relative `.lock` component with no host separators.
fn is_valid_lock_name(name: &str, path: &Utf8Path) -> bool {
    is_single_normal_component(path)
        && name.ends_with(LOCK_SUFFIX)
        && !name.contains('\\')
        && !name.contains('\0')
}

/// Whether `path` is exactly one normal component, with no root or parent.
fn is_single_normal_component(path: &Utf8Path) -> bool {
    let mut components = path.components();
    matches!(components.next(), Some(Utf8Component::Normal(_))) && components.next().is_none()
}

/// Open or create the lockfile without truncating it.
///
/// The open is non-blocking so an existing FIFO without a reader cannot hang
/// before the acquisition clock starts, and the opened descriptor must be a
/// regular file.
fn open_lockfile(dir: &Dir, name: &Utf8Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create(true);
    platform::set_nonblocking_open(&mut options);
    let file = dir.open_with(name, &options)?;
    ensure_regular_file(&file)?;
    Ok(file)
}

/// Reject a lockfile path that is not a regular file.
fn ensure_regular_file(file: &File) -> io::Result<()> {
    if file.metadata()?.is_file() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "staging lock path is not a regular file",
        ))
    }
}

/// Attempt the lock until the policy reports success or a timeout.
fn acquire_file(
    file: File,
    path: &Utf8Path,
    policy: LockPolicy,
    registration: Registration,
) -> Result<StagingLockGuard, RecipeError> {
    let started = Instant::now();
    let mut warning = ContentionWarning::new(path, policy);
    loop {
        let attempt = platform::try_flock(&file)
            .map_err(|error| RecipeError::staging_lock_io(path.as_str(), error))?;
        let elapsed = started.elapsed();
        match policy.next_step(attempt, elapsed, warning.is_emitted()) {
            LockStep::Acquired => {
                let key = registration.into_path();
                return Ok(StagingLockGuard::new(file, key));
            }
            LockStep::TimedOut { warn } => {
                // A contention seen only at the deadline never retries, so its
                // warning is owed here. The descriptor then closes, which also
                // releases a success that arrived after the deadline, and the
                // registration removes its entry.
                warning.emit(warn);
                platform::unlock(&file);
                return Err(RecipeError::staging_lock_contended(
                    path.as_str(),
                    duration_millis(elapsed),
                ));
            }
            LockStep::Retry { delay, warn } => {
                warning.emit(warn);
                thread::sleep(delay);
            }
        }
    }
}

/// The single contention warning owed by one acquisition attempt.
struct ContentionWarning<'a> {
    /// Lockfile path reported in the warning.
    path: &'a Utf8Path,
    /// Timeout reported in the warning.
    timeout: Duration,
    /// Whether the warning has already been emitted.
    emitted: bool,
}

impl<'a> ContentionWarning<'a> {
    /// Bind a warning to the acquisition it belongs to.
    const fn new(path: &'a Utf8Path, policy: LockPolicy) -> Self {
        Self {
            path,
            timeout: policy.timeout(),
            emitted: false,
        }
    }

    /// Whether the warning has been emitted already.
    const fn is_emitted(&self) -> bool {
        self.emitted
    }

    /// Emit the warning if the policy asked for it and it is still owed.
    fn emit(&mut self, warn: bool) {
        if warn && !self.emitted {
            self.emitted = true;
            warn!(
                path = self.path.as_str(),
                timeout_ms = duration_millis(self.timeout),
                "staging lock is held; contention observed"
            );
        }
    }
}

/// Convert a duration to whole milliseconds without truncation.
fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}
