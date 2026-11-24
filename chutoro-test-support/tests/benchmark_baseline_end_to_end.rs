//! End-to-end tests for the benchmark baseline helper against real Criterion.
//!
//! These tests build an isolated fixture crate with the workspace's pinned
//! Criterion 0.5.1, save a reference and a current baseline into separate
//! absolute target directories, stage the reference data into the current
//! tree, run the real `--load-baseline … --baseline …` comparison, and
//! classify the result. They are the acceptance evidence that the lane can
//! execute a comparison end to end rather than failing before it starts.
//!
//! Helpers return `Result` and every test body consumes them with a reasoned
//! `expect`, so a setup failure names its stage instead of panicking inside a
//! helper where `expect_used` is denied.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use chutoro_test_support::ci::benchmark_baseline::args::{ClassifyArgs, StageArgs};
use chutoro_test_support::ci::benchmark_baseline::compatibility;
use chutoro_test_support::ci::benchmark_baseline::manifest::{
    BaselineManifest, InventoryRequest, inventory, write_manifest,
};
use chutoro_test_support::ci::benchmark_baseline::preflight::PreflightReport;
use chutoro_test_support::ci::benchmark_baseline::runner::{
    PREFLIGHT_FILE, run_classify, run_stage,
};
use chutoro_test_support::ci::benchmark_baseline::transfer::{StageRequest, stage};
use chutoro_test_support::ci::benchmark_baseline::{
    BaselineError, BaselineVerdict, CaseClass, Outcome,
};
use mockable::DefaultEnv;
use tempfile::TempDir;

/// Convenient result type for these helpers.
type TestResult<T> = Result<T, Box<dyn Error>>;

/// Directory holding the isolated Criterion fixture.
fn fixture_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/criterion_baseline_fixture")
}

/// Directory the repository uses for build output.
fn workspace_target_directory() -> TestResult<PathBuf> {
    let parent = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("test-support must be inside the workspace")?;
    Ok(parent.join("target"))
}

/// Create a temporary root beneath the workspace target directory.
///
/// The fixture's target directories live under the workspace `target/` so a
/// dedicated `target`-shaped directory is reused rather than indexed, and so
/// a full disk is visible where the repository's build output already lives.
fn temporary_root() -> TestResult<TempDir> {
    let target = workspace_target_directory()?;
    fs::create_dir_all(&target)?;
    Ok(tempfile::Builder::new()
        .prefix("benchmark-baseline-e2e-")
        .tempdir_in(target)?)
}

/// Report a failed child command with its captured output.
fn child_failure(arguments: &[&str], output: &Output) -> String {
    format!(
        "fixture bench failed for {arguments:?}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run the fixture bench with the supplied Criterion arguments.
fn run_fixture(target_directory: &Path, slowdown: u64, arguments: &[&str]) -> TestResult<Output> {
    let output = Command::new(env!("CARGO"))
        .args(["bench", "--offline", "--locked", "--manifest-path"])
        .arg(fixture_directory().join("Cargo.toml"))
        .args(["--bench", "baseline", "--"])
        .args(arguments)
        .env("CARGO_TARGET_DIR", target_directory)
        .env_remove("CRITERION_HOME")
        .env("CHUTORO_FIXTURE_SLOWDOWN", slowdown.to_string())
        .output()?;
    if !output.status.success() {
        return Err(child_failure(arguments, &output).into());
    }

    Ok(output)
}

/// Run the fixture bench, saving `baseline` into `target_directory`.
fn save_baseline(target_directory: &Path, baseline: &str, slowdown: u64) -> TestResult<()> {
    run_fixture(
        target_directory,
        slowdown,
        &["--save-baseline", baseline, "--noplot", "--color", "never"],
    )?;
    Ok(())
}

/// Run the fixture bench with the comparison flags and filter.
fn compare_baseline(target_directory: &Path, filter: &str, slowdown: u64) -> TestResult<()> {
    run_fixture(
        target_directory,
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
fn inventory_tree(
    target_directory: &Path,
    baseline: &str,
    revision: &str,
) -> Result<BaselineManifest, BaselineError> {
    let root = target_directory.join("criterion");
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
fn write_manifests(
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
fn stage_reference(
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
fn classify(evidence: &Path, threshold: f64) -> Result<BaselineVerdict, BaselineError> {
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
fn measure_pair(
    root: &Path,
    current_slowdown: u64,
) -> TestResult<(BaselineManifest, BaselineManifest)> {
    let reference_target = root.join("reference-target");
    let current_target = root.join("current-target");
    save_baseline(&reference_target, "ci-reference", 1)?;
    save_baseline(&current_target, "ci-current", current_slowdown)?;

    Ok((
        inventory_tree(&reference_target, "ci-reference", "reference-revision")?,
        inventory_tree(&current_target, "ci-current", "current-revision")?,
    ))
}

#[test]
fn unchanged_fixture_passes_a_real_comparison() {
    let root = temporary_root().expect("temporary root must be created");
    let evidence = root.path().join("evidence");
    fs::create_dir_all(&evidence).expect("evidence directory must be created");
    let (reference, current) =
        measure_pair(root.path(), 1).expect("the fixture must measure both revisions");
    assert_eq!(reference.cases.len(), 3);
    assert!(
        reference
            .cases
            .iter()
            .all(|case| case.required_files.is_complete())
    );

    let (reference_path, current_path) =
        write_manifests(&evidence, &reference, &current).expect("manifests must be written");
    let filter = stage_reference(&evidence, &reference_path, &current_path)
        .expect("a filter must be emitted");
    assert!(filter.starts_with("^("));
    assert!(filter.ends_with(")$"));

    let current_target = root.path().join("current-target");
    compare_baseline(&current_target, &filter, 1).expect("the comparison must run");

    for case in &current.cases {
        let change = current_target
            .join("criterion")
            .join(&case.relative_directory)
            .join("change/estimates.json");
        assert!(
            change.is_file(),
            "comparison must write {}",
            change.display()
        );
    }

    let verdict = classify(&evidence, 0.05).expect("classification must succeed");
    assert_eq!(verdict.outcome, Outcome::Pass);
    assert_eq!(verdict.comparable_count, 3);
    assert!(
        verdict
            .cases
            .iter()
            .all(|case| { matches!(case.class, CaseClass::WithinNoise | CaseClass::Improved) })
    );
    let summary = verdict.to_markdown();
    assert!(summary.contains("| Outcome | pass |"), "{summary}");
}

#[test]
fn controlled_slowdown_is_detected_as_a_regression() {
    let root = temporary_root().expect("temporary root must be created");
    let evidence = root.path().join("evidence");
    fs::create_dir_all(&evidence).expect("evidence directory must be created");
    let (reference, current) =
        measure_pair(root.path(), 3).expect("the fixture must measure both revisions");

    let (reference_path, current_path) =
        write_manifests(&evidence, &reference, &current).expect("manifests must be written");
    let filter = stage_reference(&evidence, &reference_path, &current_path)
        .expect("a filter must be emitted");
    let current_target = root.path().join("current-target");
    compare_baseline(&current_target, &filter, 3).expect("the comparison must run");

    let verdict = classify(&evidence, 0.25).expect("classification must succeed");
    assert_eq!(verdict.outcome, Outcome::Regression);
    assert_eq!(verdict.regressed_ids().len(), 3);
    assert!(
        verdict
            .cases
            .iter()
            .all(|case| case.class == CaseClass::Regressed)
    );
}

#[test]
fn current_only_identifier_is_unverified_and_the_strict_filter_does_not_panic() {
    let root = temporary_root().expect("temporary root must be created");
    let evidence = root.path().join("evidence");
    fs::create_dir_all(&evidence).expect("evidence directory must be created");
    let (mut reference, current) =
        measure_pair(root.path(), 1).expect("the fixture must measure both revisions");
    let removed = reference
        .cases
        .pop()
        .expect("the fixture must register at least one case");

    let (reference_path, current_path) =
        write_manifests(&evidence, &reference, &current).expect("manifests must be written");
    let filter = stage_reference(&evidence, &reference_path, &current_path)
        .expect("a filter must be emitted");
    assert!(!filter.contains(&removed.full_id), "{filter}");
    let current_target = root.path().join("current-target");
    // `--baseline ci-reference` is strict: if the filter selected the
    // current-only identifier, Criterion would panic on its missing baseline.
    compare_baseline(&current_target, &filter, 1).expect("the comparison must run");

    let verdict = classify(&evidence, 0.05).expect("classification must succeed");
    assert_eq!(verdict.outcome, Outcome::Pass);
    assert_eq!(verdict.comparable_count, 2);
    let unverified: Vec<&str> = verdict
        .cases
        .iter()
        .filter(|case| case.class == CaseClass::NewUnverified)
        .map(|case| case.id.as_str())
        .collect();
    assert_eq!(unverified, vec![removed.full_id.as_str()]);
}

#[test]
fn staging_rejects_a_reference_tree_with_a_deleted_baseline() {
    let root = temporary_root().expect("temporary root must be created");
    let evidence = root.path().join("evidence");
    fs::create_dir_all(&evidence).expect("evidence directory must be created");
    let reference_target = root.path().join("reference-target");
    let current_target = root.path().join("current-target");
    save_baseline(&reference_target, "ci-reference", 1).expect("reference must be saved");
    save_baseline(&current_target, "ci-current", 1).expect("current must be saved");
    let reference = inventory_tree(&reference_target, "ci-reference", "reference-revision")
        .expect("reference must be inventoried");
    let current = inventory_tree(&current_target, "ci-current", "current-revision")
        .expect("current must be inventoried");
    let victim = reference
        .cases
        .first()
        .expect("the fixture must register at least one case");
    fs::remove_dir_all(
        reference
            .output_root
            .join(&victim.relative_directory)
            .join("ci-reference"),
    )
    .expect("the reference baseline directory must be removed");
    let (reference_path, current_path) =
        write_manifests(&evidence, &reference, &current).expect("manifests must be written");

    let error = run_stage(&StageArgs {
        bench: "baseline".to_owned(),
        reference_manifest: reference_path,
        current_manifest: current_path,
        evidence: evidence.clone(),
        filter_output: None,
    })
    .expect_err("a deleted reference baseline must fail staging");

    let message = error.to_string();
    assert!(message.contains("ci-reference"), "{message}");
    let report = PreflightReport::read(&evidence.join(PREFLIGHT_FILE))
        .expect("a preflight report must be written on failure");
    assert_eq!(report.outcome, Outcome::Incomplete);
    let reason = report.reason.unwrap_or_default();
    assert!(reason.contains(&victim.full_id), "{reason}");
}

#[test]
fn stage_transfer_is_exercised_directly_for_nested_identifiers() {
    let root = temporary_root().expect("temporary root must be created");
    let reference_target = root.path().join("reference-target");
    let current_target = root.path().join("current-target");
    save_baseline(&reference_target, "ci-reference", 1).expect("reference must be saved");
    save_baseline(&current_target, "ci-current", 1).expect("current must be saved");
    let reference = inventory_tree(&reference_target, "ci-reference", "reference-revision")
        .expect("reference must be inventoried");
    let current = inventory_tree(&current_target, "ci-current", "current-revision")
        .expect("current must be inventoried");
    let compatibility =
        compatibility::check(&reference, &current).expect("the two trees must be compatible");

    let outcome = stage(&StageRequest {
        reference_root: &reference.output_root,
        current_root: &current.output_root,
        reference_baseline: "ci-reference",
        comparable: &compatibility.comparable,
    })
    .expect("the transfer must succeed");

    assert_eq!(outcome.staged.len(), 3);
    assert!(
        compatibility
            .comparable
            .iter()
            .any(|case| case.full_id == "fixture/alpha/with/slash")
    );
    for staged in &outcome.staged {
        assert!(staged.destination.join("sample.json").is_file());
    }
}
