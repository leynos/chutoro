//! Fixture driver shared by the benchmark baseline end-to-end tests.
#![allow(
    dead_code,
    reason = "a test-harness module is compiled whole, and not every test uses every helper"
)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

use chutoro_test_support::ci::benchmark_baseline::args::{ClassifyArgs, StageArgs};
use chutoro_test_support::ci::benchmark_baseline::manifest::{
    BaselineManifest, InventoryRequest, inventory, write_manifest,
};
use chutoro_test_support::ci::benchmark_baseline::output_root;
use chutoro_test_support::ci::benchmark_baseline::runner::{
    PREFLIGHT_FILE, run_classify, run_stage,
};
use chutoro_test_support::ci::benchmark_baseline::{BaselineError, BaselineVerdict, Outcome};
use mockable::{DefaultEnv, MockEnv};
use tempfile::TempDir;

/// Convenient result type for these helpers.
pub type TestResult<T> = Result<T, Box<dyn Error>>;

/// Directory holding the isolated Criterion fixture.
pub fn fixture_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/criterion_baseline_fixture")
}

/// Directory the repository uses for build output.
pub fn workspace_target_directory() -> TestResult<PathBuf> {
    let parent = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("test-support must be inside the workspace")?;
    Ok(parent.join("target"))
}

/// Create a temporary root beneath the workspace target directory.
///
/// Per-run Criterion roots live under the workspace `target/` so the evidence
/// is on the same filesystem as the build output and never lands in `/tmp`.
pub fn temporary_root() -> TestResult<TempDir> {
    let target = workspace_target_directory()?;
    fs::create_dir_all(&target)?;
    Ok(tempfile::Builder::new()
        .prefix("benchmark-baseline-e2e-")
        .tempdir_in(target)?)
}

/// Build the fixture once for the whole test binary.
///
/// Criterion and the fixture take minutes to compile. Compiling them into each
/// test's own target directory would multiply that cost by the number of
/// tests, so all tests share one build and separate their Criterion output
/// roots with `CRITERION_HOME`. That also exercises the non-default output
/// root the helper must resolve. `OnceLock` makes the build happen once even
/// when the test harness runs the tests on several threads.
pub fn shared_target() -> TestResult<&'static Path> {
    static TARGET: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    let built = TARGET.get_or_init(|| {
        let workspace_target = workspace_target_directory().map_err(|error| error.to_string())?;
        let target = workspace_target.join("benchmark-baseline-fixture");
        fs::create_dir_all(&target).map_err(|error| error.to_string())?;
        let output = Command::new(env!("CARGO"))
            .args([
                "bench",
                "--offline",
                "--locked",
                "--no-run",
                "--manifest-path",
            ])
            .arg(fixture_directory().join("Cargo.toml"))
            .args(["--bench", "baseline"])
            .env("CARGO_TARGET_DIR", &target)
            .env_remove("CRITERION_HOME")
            .output()
            .map_err(|error| error.to_string())?;
        if !output.status.success() {
            return Err(child_failure(&["--no-run"], &output));
        }
        Ok(target)
    });
    match built {
        Ok(target) => Ok(target.as_path()),
        Err(message) => Err(message.clone().into()),
    }
}

/// Report a failed child command with its captured output.
pub fn child_failure(arguments: &[&str], output: &Output) -> String {
    format!(
        "fixture bench failed for {arguments:?}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run the fixture bench with the supplied Criterion arguments.
pub fn run_fixture(criterion_home: &Path, slowdown: u64, arguments: &[&str]) -> TestResult<Output> {
    let output = Command::new(env!("CARGO"))
        .args(["bench", "--offline", "--locked", "--manifest-path"])
        .arg(fixture_directory().join("Cargo.toml"))
        .args(["--bench", "baseline", "--"])
        .args(arguments)
        .env("CARGO_TARGET_DIR", shared_target()?)
        .env("CRITERION_HOME", criterion_home)
        .env("CHUTORO_FIXTURE_SLOWDOWN", slowdown.to_string())
        .output()?;
    if !output.status.success() {
        return Err(child_failure(arguments, &output).into());
    }

    Ok(output)
}

/// Run the fixture bench, saving `baseline` into `criterion_home`.
pub fn save_baseline(criterion_home: &Path, baseline: &str, slowdown: u64) -> TestResult<()> {
    run_fixture(
        criterion_home,
        slowdown,
        &["--save-baseline", baseline, "--noplot", "--color", "never"],
    )?;
    Ok(())
}

/// Run the fixture bench with the comparison flags and filter.
pub fn compare_baseline(criterion_home: &Path, filter: &str, slowdown: u64) -> TestResult<()> {
    run_fixture(
        criterion_home,
        slowdown,
        &[
            "--load-baseline",
            "ci-current",
            "--baseline",
            "ci-reference",
            "--noplot",
            "--color",
            "never",
            filter,
        ],
    )?;
    Ok(())
}

/// Inventory one baseline tree without consulting the process environment.
pub fn inventory_tree(
    criterion_home: &Path,
    baseline: &str,
    revision: &str,
) -> Result<BaselineManifest, BaselineError> {
    // Resolve the root through the helper's own precedence with
    // `CRITERION_HOME` set, so the non-default-root path is exercised rather
    // than bypassed by passing a path straight through.
    let mut env = MockEnv::new();
    let home = criterion_home.to_path_buf();
    env.expect_raw().returning(move |key| {
        if key == "CRITERION_HOME" {
            Ok(home.display().to_string())
        } else {
            Err(std::env::VarError::NotPresent)
        }
    });
    env.expect_all().returning(std::collections::HashMap::new);
    let root = output_root::resolve_workspace_output_root(criterion_home, &env)?;
    if root != criterion_home {
        return Err(BaselineError::OutputRoot {
            reason: format!(
                "the helper resolved {} instead of the supplied CRITERION_HOME {}",
                root.display(),
                criterion_home.display()
            ),
            remedy: "fix output-root precedence in the helper",
        });
    }
    let environment = BTreeMap::new();
    inventory(
        &InventoryRequest {
            workspace: &fixture_directory(),
            bench: "baseline",
            baseline,
            revision,
            output_root: &root,
            fresh_since: None,
            bench_arguments: &[
                "--noplot".to_owned(),
                "--color".to_owned(),
                "never".to_owned(),
            ],
            environment: &environment,
        },
        &DefaultEnv,
    )
}

/// Write both manifests into the evidence directory.
pub fn write_manifests(
    evidence: &Path,
    reference: &BaselineManifest,
    current: &BaselineManifest,
) -> TestResult<(PathBuf, PathBuf)> {
    let reference_path = evidence.join("reference-manifest.json");
    let current_path = evidence.join("current-manifest.json");
    write_manifest(&reference_path, reference)?;
    write_manifest(&current_path, current)?;
    Ok((reference_path, current_path))
}

/// Stage the reference data and return the emitted filter.
pub fn stage_reference(
    evidence: &Path,
    reference_path: &Path,
    current_path: &Path,
) -> TestResult<String> {
    let filter_path = evidence.join("filter.txt");
    let report = run_stage(&StageArgs {
        bench: "baseline".to_owned(),
        reference_manifest: reference_path.to_path_buf(),
        current_manifest: current_path.to_path_buf(),
        evidence: evidence.to_path_buf(),
        filter_output: Some(filter_path.clone()),
    })?;
    if report.outcome != Outcome::Pass {
        return Err(format!("staging reported outcome {:?}", report.outcome).into());
    }
    let filter = fs::read_to_string(filter_path)?;
    Ok(filter.trim().to_owned())
}

/// Classify the staged comparison.
pub fn classify(evidence: &Path, threshold: f64) -> Result<BaselineVerdict, BaselineError> {
    run_classify(&ClassifyArgs {
        preflight: evidence.join(PREFLIGHT_FILE),
        bench: "baseline".to_owned(),
        verdict: evidence.join("verdict.json"),
        threshold,
        no_reference: None,
        failed_stage: None,
        failure_reason: None,
    })
}

/// Prepare a reference and a current measurement in separate target trees.
pub fn measure_pair(
    root: &Path,
    current_slowdown: u64,
) -> TestResult<(BaselineManifest, BaselineManifest)> {
    let reference_home = root.join("reference-criterion");
    let current_home = root.join("current-criterion");
    save_baseline(&reference_home, "ci-reference", 1)?;
    save_baseline(&current_home, "ci-current", current_slowdown)?;

    Ok((
        inventory_tree(&reference_home, "ci-reference", "reference-revision")?,
        inventory_tree(&current_home, "ci-current", "current-revision")?,
    ))
}
