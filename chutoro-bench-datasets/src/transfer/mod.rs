//! Staging-file locking for resumable dataset downloads.
//!
//! Roadmap item 10.1.2 stages partial downloads inside a capability root and
//! must exclude concurrent writers to one staged artefact. This module
//! provides an advisory lock over a stable `.lock` file, built on
//! `rustix::fs::flock` and a capability-scoped `cap_std::fs_utf8::Dir`.
//!
//! The module is available with the `http` feature. It contains only the lock
//! primitive; the transfer ports, the staging slot, and the download driver
//! arrive in later 10.1.2 stages.
//!
//! # Platform support
//!
//! Locking requires `cap_std::fs_utf8::File: AsFd` and `rustix::fs::flock`.
//! It is unavailable on Windows and on the targets rustix excludes from
//! `flock` (`espidf`, `horizon`, `solaris`, `vita`, and `wasi`). On those
//! targets [`StagingLock::acquire`] returns
//! [`RecipeError::StagingLockUnsupportedPlatform`](crate::RecipeError::StagingLockUnsupportedPlatform)
//! without touching the filesystem.
//!
//! # Lock lifetime
//!
//! A [`StagingLockGuard`] owns the only descriptor for the lockfile. Dropping
//! the guard unlocks the file, removes the process-wide registry entry, and
//! closes the descriptor. The lockfile itself is never removed or renamed, so
//! a later acquisition sees the same stable pathname.
//!
//! # Filesystem requirement
//!
//! `flock` is advisory. NFS may ignore it and overlay mounts may emulate it,
//! so a staging root must live on a local filesystem. `flock` state is also
//! per open file description, so two descriptors for the same file conflict
//! while a duplicated descriptor does not; the crate never clones or hands off
//! the descriptor it holds.
//!
//! # Limitations
//!
//! - The in-process registry is keyed by the supplied root joined with the
//!   validated name, not by a canonicalized path. Two spellings of one
//!   directory are distinct keys, so callers must pass one stable root.
//! - A child that inherits the descriptor can keep the lock held after the
//!   parent exits. `cap-std` opens with close-on-exec, which covers ordinary
//!   `exec` but not an explicit descriptor handoff.
//! - A lock with no timeout would wedge every other job with no log line, so
//!   [`LockPolicy`] rejects a zero timeout and warns once on kernel-observed
//!   contention. An in-process duplicate is rejected before the kernel loop
//!   and logs its own warning instead.

mod lock;
mod platform;
mod policy;

pub use lock::{StagingLock, StagingLockGuard};
pub use policy::{
    DEFAULT_POLL_INTERVAL, LockAttempt, LockPolicy, LockPolicyError, LockStep, MIN_POLL_INTERVAL,
};
