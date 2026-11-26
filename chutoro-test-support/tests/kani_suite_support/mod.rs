//! The fixture a `kani_suite` checker case runs against.
//!
//! Each case writes a manifest and a fake `cargo kani` that prints a scripted
//! regular-format run into a directory of its own. The fake is a real process,
//! so a case exercises the whole path the Makefile uses: argument validation,
//! the delegated environment, streaming, parsing, and the comparison.
//!
//! The fixture's version banner is read from `tools/kani/VERSION` rather than
//! restated, so a version bump moves the fixture with the pin instead of
//! leaving a second copy to drift.
//!
//! This module lives under `tests/kani_suite_support/` so Cargo does not treat
//! it as a standalone test crate; `kani_suite_cli.rs` pulls it in with
//! `#[path]`, which is how the neighbouring contract suites carry shared
//! support too.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use rstest::fixture;

/// The Kani version pin the fixture banners must carry.
pub const VERSION: &str = include_str!("../../../tools/kani/VERSION");

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
/// delegated environment arrived by writing the value it saw. It is a real
/// script so its shell can be read, edited and run on its own rather than
/// through a Rust string literal.
const FAKE: &str = include_str!("../fixtures/kani_suite_fake.sh");

/// A reason a fixture could not be prepared.
#[derive(Debug)]
pub struct FixtureError(pub String);

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
pub fn required<T, E: std::fmt::Display>(
    result: Result<T, E>,
    what: &str,
) -> Result<T, FixtureError> {
    result.map_err(|error| FixtureError(format!("{what}: {error}")))
}

/// One prepared checker case: the manifest path and the fake verifier.
pub struct Case {
    /// The directory holding the fixture; removed when the case ends.
    pub directory: std::path::PathBuf,
    /// The manifest file the checker is given.
    pub manifest: std::path::PathBuf,
    /// The fake `cargo kani`, which the checker runs as its delegated command.
    pub verifier: std::path::PathBuf,
}

impl Case {
    /// Write the fixture into a directory this case creates exclusively.
    ///
    /// `create_dir` rather than `create_dir_all`: an interrupted run leaves its
    /// directory behind, and a later run whose process id and sequence happen
    /// to repeat would otherwise adopt that directory, complete with the marker
    /// an earlier case wrote, and then delete it in `Drop`. Exclusive creation
    /// fails instead, and the test reports a stale directory rather than
    /// behaving differently for a reason that has nothing to do with the
    /// checker.
    ///
    /// The `Case` is built before anything is written, so a failed write still
    /// returns a value whose `Drop` removes the partially written directory.
    /// Building it afterwards would leak every failed setup into the temporary
    /// directory, because no owner would exist to clean it up.
    pub fn create(name: &str) -> Result<Self, FixtureError> {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "chutoro-kani-suite-cli-{}-{sequence}-{name}",
            std::process::id()
        ));
        let manifest = directory.join("harnesses.toml");
        let verifier = directory.join("cargo-kani-fake");
        let case = Self {
            directory,
            manifest,
            verifier,
        };
        required(
            fs::create_dir(&case.directory),
            "create the fixture directory exclusively",
        )?;
        required(
            fs::write(&case.manifest, MANIFEST),
            "write the fixture manifest",
        )?;
        required(fs::write(&case.verifier, FAKE), "write the fake verifier")?;
        set_executable(&case.verifier)?;
        Ok(case)
    }

    /// Run the checker with the supplied extra arguments and selectors.
    pub fn run(
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
pub fn case() -> Result<Case, FixtureError> {
    Case::create("default")
}
