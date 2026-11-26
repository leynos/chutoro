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

#[path = "kani_suite_support/mod.rs"]
mod support;

use std::process::Command;

use rstest::rstest;
use support::{Case, FixtureError, VERSION, case};

/// The captured stream a case's expectation is checked against.
#[derive(Debug, Clone, Copy)]
enum Stream {
    /// Standard output, where a successful run prints its summary.
    Out,
    /// Standard error, where the checker prints its diagnostics.
    Err,
}

impl Stream {
    /// Return this stream's captured text.
    fn text(self, output: &std::process::Output) -> String {
        let captured = match self {
            Self::Out => &output.stdout,
            Self::Err => &output.stderr,
        };
        String::from_utf8_lossy(captured).into_owned()
    }
}

/// Require the checker to have reached `expected` status and named `needle`.
///
/// Both directions share this check: they differ only in which stream carries
/// the text and whether success is wanted, and a second copy of the body would
/// eventually describe only one of them.
fn expect_outcome(
    output: &std::process::Output,
    expected: bool,
    stream: Stream,
    needle: &str,
) -> Result<(), String> {
    if output.status.success() != expected {
        let wanted = if expected { "success" } else { "failure" };
        return Err(format!(
            "expected {wanted}, got {}: {}",
            output.status,
            stream.text(output)
        ));
    }
    let text = stream.text(output);
    if !text.contains(needle) {
        return Err(format!("expected {needle:?} in the output, got {text}"));
    }
    Ok(())
}

/// Require the checker to have failed, naming `needle` in its diagnostics.
///
/// A `Result` rather than asserts, so the `#[rstest]` bodies can return it:
/// a `panic_in_result_fn` denial applies to any test that propagates a
/// fixture, and the checks belong beside the behaviour they describe rather
/// than in a panicking helper.
fn rejected(output: &std::process::Output, needle: &str) -> Result<(), String> {
    expect_outcome(output, false, Stream::Err, needle)
}

/// Require the checker to have succeeded, with a `needle` in its output.
fn accepted(output: &std::process::Output, needle: &str) -> Result<(), String> {
    expect_outcome(output, true, Stream::Out, needle)
}

#[rstest]
fn accepts_an_exact_successful_run(case: Result<Case, FixtureError>) -> Result<(), String> {
    let fixture = case.map_err(|error| error.to_string())?;
    let output = fixture
        .run(&[], &["proofs::verify_one"], &[])
        .map_err(|error| error.to_string())?;
    accepted(&output, "verified 1 harness(es) for fixture (fast)")?;
    // The delegated verifier's own banner proves the run was streamed, not
    // merely parsed from a captured file.
    accepted(&output, &format!("Kani Rust Verifier {}", VERSION.trim()))
}

/// One rejection case: what the fake prints, and the diagnostic it must earn.
#[derive(Debug, Clone, Copy)]
struct Rejection {
    /// The `FAKE_KANI_MODE` the fake verifier runs.
    mode: &'static str,
    /// The selectors the checker is asked for.
    selectors: &'static [&'static str],
    /// The diagnostic fragment that must name the defect.
    needle: &'static str,
}

#[rstest]
#[case::missing(Rejection {
    mode: "missing",
    selectors: &["proofs::verify_one", "proofs::verify_two"],
    needle: "did not complete these expected harnesses",
})]
#[case::unexpected(Rejection {
    mode: "unexpected",
    selectors: &["proofs::verify_one"],
    needle: "outside the requested scope",
})]
#[case::duplicate(Rejection {
    mode: "duplicate",
    selectors: &["proofs::verify_one"],
    needle: "more than once",
})]
#[case::failed(Rejection {
    mode: "failed",
    selectors: &["proofs::verify_one"],
    needle: "failed or undetermined outcome",
})]
#[case::incomplete(Rejection {
    mode: "incomplete",
    selectors: &["proofs::verify_one"],
    needle: "never reported an outcome",
})]
#[case::no_summary(Rejection {
    mode: "no-summary",
    selectors: &["proofs::verify_one"],
    needle: "carries no manual harness summary",
})]
/// Every way a run can fail the expected set is reported as its own defect.
fn rejects_a_run_that_does_not_match_the_expected_set(
    case: Result<Case, FixtureError>,
    #[case] rejection: Rejection,
) -> Result<(), String> {
    let fixture = case.map_err(|error| error.to_string())?;
    let output = fixture
        .run(
            &[],
            rejection.selectors,
            &[("FAKE_KANI_MODE", rejection.mode)],
        )
        .map_err(|error| error.to_string())?;
    rejected(&output, rejection.needle)
}

#[rstest]
fn propagates_a_nonzero_delegated_status(case: Result<Case, FixtureError>) -> Result<(), String> {
    let fixture = case.map_err(|error| error.to_string())?;
    let output = fixture
        .run(
            &[],
            &["proofs::verify_one"],
            &[("FAKE_KANI_MODE", "silent"), ("FAKE_KANI_STATUS", "7")],
        )
        .map_err(|error| error.to_string())?;
    rejected(&output, "failed with exit status: 7")
}

#[rstest]
fn rejects_a_quiet_run_before_starting_it(case: Result<Case, FixtureError>) -> Result<(), String> {
    let fixture = case.map_err(|error| error.to_string())?;
    let marker = fixture.directory.join("invoked");
    let output = fixture
        .run(
            &["--quiet"],
            &["proofs::verify_one"],
            &[("FAKE_KANI_MARKER", &marker.to_string_lossy())],
        )
        .map_err(|error| error.to_string())?;
    rejected(&output, "--quiet")?;
    if marker.exists() {
        return Err(String::from(
            "the checker must refuse the command before running it",
        ));
    }
    Ok(())
}

#[rstest]
fn rejects_a_selector_outside_the_manifest_scope(
    case: Result<Case, FixtureError>,
) -> Result<(), String> {
    let fixture = case.map_err(|error| error.to_string())?;
    let output = fixture
        .run(&[], &["proofs::verify_unlisted"], &[])
        .map_err(|error| error.to_string())?;
    rejected(&output, "outside the fast scope")
}

#[rstest]
fn applies_a_leading_assignment_to_the_delegated_process(
    case: Result<Case, FixtureError>,
) -> Result<(), String> {
    let fixture = case.map_err(|error| error.to_string())?;
    let output = fixture
        .run(
            &[],
            &["proofs::verify_one", "proofs::verify_two"],
            &[("FAKE_KANI_MODE", "two")],
        )
        .map_err(|error| error.to_string())?;
    // The assignment selects the two-harness output, so a checker that failed
    // to pass it would report a missing harness instead.
    accepted(&output, "verified 2 harness(es) for fixture (fast)")
}

#[rstest]
fn rejects_a_command_that_selects_another_package(
    case: Result<Case, FixtureError>,
) -> Result<(), String> {
    let fixture = case.map_err(|error| error.to_string())?;
    let manifest = fixture.manifest.to_string_lossy().into_owned();
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
        .arg(&fixture.verifier)
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
    let output = raw.map_err(|error| error.to_string())?;
    rejected(&output, "must select exactly package")
}
