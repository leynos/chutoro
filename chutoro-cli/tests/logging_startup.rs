//! Process-level checks for the CLI's logging start-up.
//!
//! Logging is installed once per process, so only a freshly spawned binary
//! shows what an operator sees on `stderr` before any command runs.

use std::{
    io,
    process::{Command, Output},
};

/// Run the CLI binary Cargo built for this package with `args`.
fn run_cli(args: &[&str]) -> io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_chutoro-cli"))
        .args(args)
        .env_remove("RUST_LOG")
        .env_remove("CHUTORO_LOG_FORMAT")
        .output()
}

/// A standalone process owns both the `tracing` dispatcher and the `log`
/// bridge, so start-up must not report a conflict with another logger.
#[test]
fn help_starts_without_a_logging_conflict_warning() {
    let output = run_cli(&["--help"]).expect("the chutoro-cli binary should start");

    assert!(output.status.success(), "--help should succeed");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("structured logging already configured"),
        "unexpected logging conflict on stderr: {stderr}"
    );
}
