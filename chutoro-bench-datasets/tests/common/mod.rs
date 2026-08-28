//! Shared assertion helpers for `chutoro-bench-datasets` integration tests.
//!
//! This module lives in a `tests/common` subdirectory (rather than a bare
//! `tests/common.rs`) so Cargo does not treat it as its own standalone test
//! crate; consumers pull it in with `mod common;`.

/// Asserts that `$expr` evaluates to `Err` and yields the wrapped error,
/// panicking at the call site with the supplied message if it does not.
///
/// Implemented as a macro (rather than a helper function) so a failed
/// assertion's panic location points at the calling test, matching the
/// diagnostics produced by the `let Err(error) = ... else { panic!(...) };`
/// idiom it replaces.
macro_rules! expect_err {
    ($expr:expr, $($panic_arg:tt)+) => {
        match $expr {
            Err(error) => error,
            Ok(_) => panic!($($panic_arg)+),
        }
    };
}

pub(crate) use expect_err;
