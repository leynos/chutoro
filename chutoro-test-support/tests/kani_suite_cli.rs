//! Behavioural tests for the `kani_suite` checker binary.
//!
//! Each case writes a manifest and a fake `cargo kani` that prints a scripted
//! regular-format run. The fake is a real process, so the tests exercise the
//! whole path the Makefile uses: argument validation, the delegated
//! environment, streaming, parsing, and the comparison.
//!
//! The fixture's version banner is read from `tools/kani/VERSION` rather than
//! restated, so a version bump moves the fixture with the pin instead of
//! leaving a second copy to drift.
//!
//! The fake verifier is a POSIX shell script marked executable, so the whole
//! integration test is gated on Unix. A non-Unix host gets no tests rather
//! than a fixture that cannot start: an unconditional panic in a shared
//! fixture would fail every case with a message about the harness instead of
//! about the checker.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use rstest::{fixture, rstest};

/// The Kani version pin the fixture banners must carry.
const VERSION: &str = include_str!("../../tools/kani/VERSION");

/// Distinguishes fixtures created concurrently by the test harness.
static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

/// The manifest used by every case: one package, two fast harnesses.
const MANIFEST: &str = concat!(
    "version = 1\n",
    "[[harness]]\n",
    "package = \"fixture\"\n",
    "pretty_name = \"proofs::verify_one\"\n",
    "tier = \"fast\"\n",
    "[[harness]]\n",
    "package = \"fixture\"\n",
    "pretty_name = \"proofs::verify_two\"\n",
    "tier = \"fast\"\n",
);

/// The fake verifier. `FAKE_KANI_MODE` selects the output and
/// `FAKE_KANI_STATUS` the exit status; a successful mode also proves the
/// delegated environment arrived by writing the value it saw.
const FAKE: &str = r#"#!/bin/sh
if [ -n "$FAKE_KANI_MARKER" ]; then
  printf '%s\n' "$FAKE_KANI_SELECTOR" >> "$FAKE_KANI_MARKER"
fi
printf 'Kani Rust Verifier %s (cargo plugin)\n' "$FAKE_KANI_VERSION"
case "${FAKE_KANI_MODE:-ok}" in
  ok)
    printf 'Checking harness proofs::verify_one...\n'
    printf 'SUMMARY:\n ** 0 of 3 failed\n\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 1 successfully verified harnesses, 0 failures, 1 total.\n'
    ;;
  two)
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Checking harness proofs::verify_two...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 2 successfully verified harnesses, 0 failures, 2 total.\n'
    ;;
  missing)
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 1 successfully verified harnesses, 0 failures, 1 total.\n'
    ;;
  unexpected)
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Checking harness proofs::verify_unlisted...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 2 successfully verified harnesses, 0 failures, 2 total.\n'
    ;;
  duplicate)
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 2 successfully verified harnesses, 0 failures, 2 total.\n'
    ;;
  failed)
    printf 'Checking harness proofs::verify_one...\n'
    printf 'SUMMARY:\n ** 1 of 3 failed\n\nVERIFICATION:- FAILED\n'
    printf 'Manual Harness Summary:\n'
    printf 'Verification failed for - proofs::verify_one\n'
    printf 'Complete - 0 successfully verified harnesses, 1 failures, 1 total.\n'
    ;;
  incomplete)
    printf 'Checking harness proofs::verify_one...\n'
    ;;
  silent)
    ;;
esac
exit "${FAKE_KANI_STATUS:-0}"
"#;

/// A reason a fixture could not be prepared.
#[derive(Debug)]
struct FixtureError(String);

impl std::fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FixtureError {}

/// Return a fallible setup step's value, or a named fixture error.
///
/// The helpers return errors rather than panicking, as the repository's
/// error-handling policy requires of shared fixtures. Each `#[rstest]` body
/// consumes one with a reasoned `.expect(...)`, which the test-only exemption
/// covers and which keeps the failure at the test boundary.
fn required<T, E: std::fmt::Display>(result: Result<T, E>, what: &str) -> Result<T, FixtureError> {
    result.map_err(|error| FixtureError(format!("{what}: {error}")))
}

/// One prepared checker case: the manifest path and the fake verifier.
struct Case {
    /// The directory holding the fixture; removed when the case ends.
    directory: PathBuf,
    /// The manifest file the checker is given.
    manifest: PathBuf,
    /// The fake `cargo kani`, which the checker runs as its delegated command.
    verifier: PathBuf,
}

impl Case {
    /// Write the fixture into a directory named after the test.
    fn create(name: &str) -> Result<Self, FixtureError> {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "chutoro-kani-suite-cli-{}-{sequence}-{name}",
            std::process::id()
        ));
        required(
            fs::create_dir_all(&directory),
            "create the fixture directory",
        )?;
        let manifest = directory.join("harnesses.toml");
        required(fs::write(&manifest, MANIFEST), "write the fixture manifest")?;
        let verifier = directory.join("cargo-kani-fake");
        required(fs::write(&verifier, FAKE), "write the fake verifier")?;
        set_executable(&verifier)?;
        Ok(Self {
            directory,
            manifest,
            verifier,
        })
    }

    /// Run the checker with the supplied extra arguments and selectors.
    fn run(
        &self,
        extra: &[&str],
        selectors: &[&str],
        environment: &[(&str, &str)],
    ) -> Result<std::process::Output, FixtureError> {
        let mut command = Command::new(env!("CARGO_BIN_EXE_kani_suite"));
        command
            .arg("--manifest")
            .arg(&self.manifest)
            .arg("--package")
            .arg("fixture")
            .arg("--scope")
            .arg("fast")
            .arg("--")
            .args(
                environment
                    .iter()
                    .map(|(name, value)| format!("{name}={value}")),
            )
            .arg(&self.verifier)
            .arg("kani")
            .arg("-p")
            .arg("fixture")
            .arg("--lib")
            .arg("--exact")
            .args(extra);
        for selector in selectors {
            command.arg("--harness").arg(selector);
        }
        command.env("FAKE_KANI_VERSION", VERSION.trim());
        required(command.output(), "run the checker")
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        if let Err(_error) = fs::remove_dir_all(&self.directory) {
            // A leaked fixture directory is untidy; it must not fail the test.
        }
    }
}

/// Mark a script executable.
///
/// The whole test module is Unix-only, so a Unix permission change is the only
/// shape this needs. A host without it gets no test run at all rather than a
/// fixture that cannot start.
fn set_executable(path: &Path) -> Result<(), FixtureError> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = required(fs::metadata(path), "stat the fake")?.permissions();
    permissions.set_mode(0o755);
    required(
        fs::set_permissions(path, permissions),
        "mark the fake executable",
    )
}

/// A case with the default fixture.
#[fixture]
fn case() -> Case {
    Case::create("default").expect("the checker fixture must be prepared")
}

/// Assert the checker failed and its diagnostics name `needle`.
fn assert_rejected(output: &std::process::Output, needle: &str) {
    assert!(
        !output.status.success(),
        "expected failure, got success: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(needle),
        "expected the diagnostic to name {needle:?}, got {stderr}"
    );
}

#[rstest]
fn accepts_an_exact_successful_run(case: Case) {
    let output = case
        .run(&[], &["proofs::verify_one"], &[])
        .expect("the checker must run");
    assert!(
        output.status.success(),
        "expected success, got {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("verified 1 harness(es) for fixture (fast)"),
        "expected the checker's summary, got {stdout}"
    );
    assert!(
        stdout.contains(&format!("Kani Rust Verifier {}", VERSION.trim())),
        "the delegated verifier's banner must be streamed: {stdout}"
    );
}

#[rstest]
fn rejects_a_missing_harness(case: Case) {
    let output = case
        .run(
            &[],
            &["proofs::verify_one", "proofs::verify_two"],
            &[("FAKE_KANI_MODE", "missing")],
        )
        .expect("the checker must run");
    assert_rejected(&output, "did not complete these expected harnesses");
}

#[rstest]
fn rejects_an_unexpected_harness(case: Case) {
    let output = case
        .run(
            &[],
            &["proofs::verify_one"],
            &[("FAKE_KANI_MODE", "unexpected")],
        )
        .expect("the checker must run");
    assert_rejected(&output, "outside the requested scope");
}

#[rstest]
fn rejects_a_duplicate_harness(case: Case) {
    let output = case
        .run(
            &[],
            &["proofs::verify_one"],
            &[("FAKE_KANI_MODE", "duplicate")],
        )
        .expect("the checker must run");
    assert_rejected(&output, "more than once");
}

#[rstest]
fn rejects_a_failed_harness(case: Case) {
    let output = case
        .run(
            &[],
            &["proofs::verify_one"],
            &[("FAKE_KANI_MODE", "failed")],
        )
        .expect("the checker must run");
    assert_rejected(&output, "failed or undetermined outcome");
}

#[rstest]
fn rejects_an_incomplete_block(case: Case) {
    let output = case
        .run(
            &[],
            &["proofs::verify_one"],
            &[("FAKE_KANI_MODE", "incomplete")],
        )
        .expect("the checker must run");
    assert_rejected(&output, "never reported an outcome");
}

#[rstest]
fn propagates_a_nonzero_delegated_status(case: Case) {
    let output = case
        .run(
            &[],
            &["proofs::verify_one"],
            &[("FAKE_KANI_MODE", "silent"), ("FAKE_KANI_STATUS", "7")],
        )
        .expect("the checker must run");
    assert_rejected(&output, "failed with exit status: 7");
}

#[rstest]
fn rejects_a_quiet_run_before_starting_it(case: Case) {
    let marker = case.directory.join("invoked");
    let output = case
        .run(
            &["--quiet"],
            &["proofs::verify_one"],
            &[("FAKE_KANI_MARKER", &marker.to_string_lossy())],
        )
        .expect("the checker must run");
    assert_rejected(&output, "--quiet");
    assert!(
        !marker.exists(),
        "the checker must refuse the command before running it"
    );
}

#[rstest]
fn rejects_a_selector_outside_the_manifest_scope(case: Case) {
    let output = case
        .run(&[], &["proofs::verify_unlisted"], &[])
        .expect("the checker must run");
    assert_rejected(&output, "outside the fast scope");
}

#[rstest]
fn applies_a_leading_assignment_to_the_delegated_process(case: Case) {
    let output = case
        .run(
            &[],
            &["proofs::verify_one", "proofs::verify_two"],
            &[("FAKE_KANI_MODE", "two")],
        )
        .expect("the checker must run");
    assert!(
        output.status.success(),
        "expected the delegated environment assignment to select the two-harness \
         output, got {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[rstest]
fn rejects_a_command_that_selects_another_package(case: Case) {
    let manifest = case.manifest.to_string_lossy().into_owned();
    let raw = Command::new(env!("CARGO_BIN_EXE_kani_suite"))
        .args([
            "--manifest",
            manifest.as_str(),
            "--package",
            "fixture",
            "--scope",
            "fast",
            "--",
        ])
        .arg(&case.verifier)
        .args([
            "kani",
            "-p",
            "another-package",
            "--lib",
            "--exact",
            "--harness",
            "proofs::verify_one",
        ])
        .output();
    let output = raw.expect("the checker must run");
    assert_rejected(&output, "must select exactly package");
}
