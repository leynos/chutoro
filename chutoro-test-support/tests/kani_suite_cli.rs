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
/// The fake verifier, kept as a real script so its shell can be read,
/// edited and run on its own rather than through a Rust string literal.
const FAKE: &str = include_str!("fixtures/kani_suite_fake.sh");

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
///
/// The fixture returns a `Result`, as the repository's other fallible fixtures
/// do, and each `#[rstest]` body propagates it with `?`. That keeps setup
/// failure out of the fixture, where neither `expect_used` nor a panicking
/// fallback is permitted.
#[fixture]
fn case() -> Result<Case, FixtureError> {
    Case::create("default")
}

/// Require the checker to have failed, naming `needle` in its diagnostics.
///
/// A `Result` rather than asserts, so the `#[rstest]` bodies can return it:
/// a `panic_in_result_fn` denial applies to any test that propagates a
/// fixture, and the checks belong beside the behaviour they describe rather
/// than in a panicking helper.
fn rejected(output: &std::process::Output, needle: &str) -> Result<(), String> {
    if output.status.success() {
        return Err(format!(
            "expected failure, got success: {}",
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains(needle) {
        return Err(format!(
            "expected the diagnostic to name {needle:?}, got {stderr}"
        ));
    }
    Ok(())
}

/// Require the checker to have succeeded, with a `needle` in its output.
fn accepted(output: &std::process::Output, needle: &str) -> Result<(), String> {
    if !output.status.success() {
        return Err(format!(
            "expected success, got {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains(needle) {
        return Err(format!("expected {needle:?} in the output, got {stdout}"));
    }
    Ok(())
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
