//! Platform bindings for the staging lock.
//!
//! `flock` and `cap-std`'s `AsFd` are both absent on some targets, so every
//! target-specific item lives here behind one predicate. The rest of the lock
//! adapter is written once against this module's surface.

use std::io;

use cap_std::fs_utf8::{File, OpenOptions};

use super::policy::LockAttempt;

/// Whether this target exposes `cap-std`'s `AsFd` and `rustix::fs::flock`.
///
/// `cap_std::fs_utf8::File` implements `AsFd` only on non-Windows targets, and
/// rustix omits `flock` on the listed systems. The same predicate gates the
/// module below and the `rustix` dependency's target table.
pub const FLOCK_SUPPORTED: bool = cfg!(not(any(
    windows,
    target_os = "espidf",
    target_os = "horizon",
    target_os = "solaris",
    target_os = "vita",
    target_os = "wasi"
)));

/// Ask the open to return instead of waiting on a reader-less FIFO.
#[cfg(not(any(
    windows,
    target_os = "espidf",
    target_os = "horizon",
    target_os = "solaris",
    target_os = "vita",
    target_os = "wasi"
)))]
pub fn set_nonblocking_open(options: &mut OpenOptions) {
    use cap_std::fs_utf8::OpenOptionsExt;

    if let Ok(flag) = i32::try_from(rustix::fs::OFlags::NONBLOCK.bits()) {
        options.custom_flags(flag);
    }
}

/// Leave the options unchanged on a target that never reaches the open.
#[cfg(any(
    windows,
    target_os = "espidf",
    target_os = "horizon",
    target_os = "solaris",
    target_os = "vita",
    target_os = "wasi"
))]
pub fn set_nonblocking_open(_options: &mut OpenOptions) {}

/// Perform one non-blocking exclusive `flock` attempt.
#[cfg(not(any(
    windows,
    target_os = "espidf",
    target_os = "horizon",
    target_os = "solaris",
    target_os = "vita",
    target_os = "wasi"
)))]
pub fn try_flock(file: &File) -> io::Result<LockAttempt> {
    use std::os::fd::AsFd;

    use rustix::{
        fs::{FlockOperation, flock},
        io::Errno,
    };

    match flock(file.as_fd(), FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => Ok(LockAttempt::Acquired),
        Err(error) if error == Errno::WOULDBLOCK => Ok(LockAttempt::Contended),
        Err(error) if error == Errno::INTR => Ok(LockAttempt::Interrupted),
        Err(error) => Err(error.into()),
    }
}

/// Report an unsupported target without touching the filesystem.
#[cfg(any(
    windows,
    target_os = "espidf",
    target_os = "horizon",
    target_os = "solaris",
    target_os = "vita",
    target_os = "wasi"
))]
pub fn try_flock(_file: &File) -> io::Result<LockAttempt> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// Release the advisory lock held by `file`.
#[cfg(not(any(
    windows,
    target_os = "espidf",
    target_os = "horizon",
    target_os = "solaris",
    target_os = "vita",
    target_os = "wasi"
)))]
pub fn unlock(file: &File) {
    use std::os::fd::AsFd;

    use rustix::fs::{FlockOperation, flock};
    use tracing::debug;

    if let Err(error) = flock(file.as_fd(), FlockOperation::Unlock) {
        debug!(
            error = %error,
            "unable to unlock staging file; closing the descriptor releases the lock"
        );
    }
}

/// Release nothing on a target without an `flock` binding.
#[cfg(any(
    windows,
    target_os = "espidf",
    target_os = "horizon",
    target_os = "solaris",
    target_os = "vita",
    target_os = "wasi"
))]
pub fn unlock(_file: &File) {}
