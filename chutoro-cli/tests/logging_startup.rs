//! Process-level checks for the CLI's logging start-up.
//!
//! Logging is installed once per process, so only a freshly spawned binary
//! shows what an operator sees on `stderr` before any command runs.

use std::{
    io::{self, Write},
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

/// A successful run prints its summary on `stdout` and nothing on `stderr`
/// unless the operator asks for diagnostics with `RUST_LOG`.
#[test]
fn successful_run_is_quiet_on_stderr_by_default() {
    let mut corpus = tempfile::Builder::new()
        .prefix("words")
        .suffix(".txt")
        .tempfile()
        .expect("a temporary corpus should be created");
    corpus
        .write_all(b"apple\napply\napples\nappel\n")
        .expect("the corpus should be written");
    let path = corpus
        .path()
        .to_str()
        .expect("the temporary path should be UTF-8");

    let output = run_cli(&["run", "--min-cluster-size", "2", "text", path])
        .expect("the chutoro-cli binary should start");

    assert!(output.status.success(), "the run should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("data source: words"),
        "unexpected stdout: {stdout}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.is_empty(),
        "unexpected diagnostics on stderr: {stderr}"
    );
}
