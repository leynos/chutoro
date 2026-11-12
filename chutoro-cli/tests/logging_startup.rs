//! Process-level checks for the CLI's logging start-up.
//!
//! Logging is installed once per process, so only a freshly spawned binary
//! shows what an operator sees on `stderr` before any command runs.

use std::{
    io::{self, Write},
    process::{Command, Output},
};

/// Text corpus the `run text` command clusters without external fixtures.
const CORPUS: &[u8] = b"alpha\nbeta\ngamma\n";

/// Message the CLI emits once a command has completed.
const COMPLETION_EVENT: &str = "command completed";

/// Diagnostic printed when logging initialization finds a taken global slot.
const CONFLICT_WARNING: &str = "structured logging already configured";

/// Run the CLI binary Cargo built for this package with `args` and `envs`.
///
/// The logging environment is cleared first so each case observes only the
/// settings it passes.
fn run_cli_with_env(args: &[&str], envs: &[(&str, &str)]) -> io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_chutoro-cli"))
        .args(args)
        .env_remove("RUST_LOG")
        .env_remove("CHUTORO_LOG_FORMAT")
        .envs(envs.iter().copied())
        .output()
}

/// Run the CLI binary Cargo built for this package with `args`.
fn run_cli(args: &[&str]) -> io::Result<Output> {
    run_cli_with_env(args, &[])
}

/// Run the `text` command over [`CORPUS`] with the supplied logging settings.
fn run_corpus(envs: &[(&str, &str)]) -> io::Result<Output> {
    let mut corpus = tempfile::NamedTempFile::new()?;
    corpus.write_all(CORPUS)?;
    corpus.flush()?;
    let path = corpus.path().to_string_lossy().into_owned();
    let args = [
        "run",
        "--min-cluster-size",
        "1",
        "text",
        path.as_str(),
        "--metric",
        "levenshtein",
    ];
    run_cli_with_env(&args, envs)
}

/// A standalone process owns both the `tracing` dispatcher and the `log`
/// bridge, so start-up must not report a conflict with another logger.
#[test]
fn help_starts_without_a_logging_conflict_warning() {
    let output = run_cli(&["--help"]).expect("the chutoro-cli binary should start");

    assert!(output.status.success(), "--help should succeed");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains(CONFLICT_WARNING),
        "unexpected logging conflict on stderr: {stderr}"
    );
}

/// A completed command must report its event on `stderr`, which proves the
/// subscriber is installed and honours `RUST_LOG` instead of merely staying
/// quiet.
#[test]
fn a_completed_run_logs_its_event() {
    let output = run_corpus(&[("RUST_LOG", "info")]).expect("the chutoro-cli binary should start");

    assert!(output.status.success(), "the run command should succeed");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(COMPLETION_EVENT),
        "expected the completion event on stderr: {stderr}"
    );
    assert!(
        !stderr.contains(CONFLICT_WARNING),
        "unexpected logging conflict on stderr: {stderr}"
    );
}

/// Selecting JSON output must produce the same event as structured data,
/// which proves `CHUTORO_LOG_FORMAT` reaches the installed formatter.
#[test]
fn a_completed_run_logs_json_when_requested() {
    let output = run_corpus(&[("RUST_LOG", "info"), ("CHUTORO_LOG_FORMAT", "json")])
        .expect("the chutoro-cli binary should start");

    assert!(output.status.success(), "the run command should succeed");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let json_event = format!("\"message\":\"{COMPLETION_EVENT}\"");
    assert!(
        stderr.contains(&json_event),
        "expected a JSON completion event on stderr: {stderr}"
    );
    assert!(
        !stderr.contains(CONFLICT_WARNING),
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
