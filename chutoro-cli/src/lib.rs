//! Support library for the chutoro CLI binary.
//!
//! Re-exports the CLI module so doctests and integration tests can exercise the
//! command pipeline without forking a subprocess.

pub mod cli;
