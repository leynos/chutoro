//! Behavioural tests for the benchmark baseline helper binary.
//!
//! Each case builds a synthetic Criterion tree, runs the real binary with a
//! controlled environment, and asserts the documented exit code together with
//! the diagnostic it printed and the evidence it wrote. The exit-code mapping
//! is the contract the workflow relies on to distinguish a regression from a
//! lane defect.

#[path = "benchmark_baseline_cli/support.rs"]
mod support;

use support::{
    Sandbox, binary, complete, exit_code, read, run, run_stage, sandbox, signal_description, stderr,
};

use std::fs;
use std::path::{Path, PathBuf};

use chutoro_test_support::ci::benchmark_baseline::manifest::read_manifest;
use chutoro_test_support::ci::benchmark_baseline::{BaselineVerdict, Outcome, RequiredFiles};
use rstest::rstest;

#[rstest]
fn usage_error_exits_sixty_four(binary: PathBuf) {
    let output = run(&binary, &[], &[]).expect("helper binary must run");

    assert_eq!(
        exit_code(&output),
        Some(64),
        "{}",
        signal_description(&output)
    );
    assert!(
        stderr(&output).contains("no subcommand"),
        "{}",
        stderr(&output)
    );
}

#[rstest]
fn help_exits_zero(binary: PathBuf) {
    let output = run(&binary, &["--help"], &[]).expect("helper binary must run");

    assert_eq!(
        exit_code(&output),
        Some(0),
        "{}",
        signal_description(&output)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("inventory"), "{stdout}");
    assert!(stdout.contains("no_reference"), "{stdout}");
}

#[rstest]
fn missing_reference_sample_exits_incomplete_and_names_the_path(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let workspace = sandbox.expect("sandbox must be created");
    let incomplete = RequiredFiles {
        sample: false,
        ..complete()
    };
    let output =
        run_stage(&binary, &workspace, ("0.5.1", incomplete), "0.5.1").expect("staging must run");

    assert_eq!(
        exit_code(&output),
        Some(2),
        "{}",
        signal_description(&output)
    );
    let message = stderr(&output);
    assert!(message.contains("fixture/alpha"), "{message}");
    assert!(message.contains("sample.json"), "{message}");
    assert!(message.contains("ci-reference"), "{message}");
}

#[rstest]
fn version_mismatch_exits_incompatible_and_names_both_versions(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let workspace = sandbox.expect("sandbox must be created");
    let output =
        run_stage(&binary, &workspace, ("0.4.0", complete()), "0.5.1").expect("staging must run");

    assert_eq!(
        exit_code(&output),
        Some(3),
        "{}",
        signal_description(&output)
    );
    let message = stderr(&output);
    assert!(message.contains("0.4.0"), "{message}");
    assert!(message.contains("0.5.1"), "{message}");
}

#[rstest]
fn inventory_honours_a_non_default_criterion_home(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let workspace = sandbox.expect("sandbox must be created");
    let custom_root = workspace.evidence.join("custom-criterion");
    Sandbox::write_baseline(&custom_root, "fixture/alpha", "ci-reference")
        .expect("custom criterion tree must be written");
    let manifest_path = workspace.evidence.join("reference-manifest.json");
    let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));

    let output = run(
        &binary,
        &[
            "inventory",
            "--workspace",
            &crate_directory.display().to_string(),
            "--bench",
            "hnsw",
            "--baseline",
            "ci-reference",
            "--revision",
            "abc",
            "--output",
            &manifest_path.display().to_string(),
            "--bench-argument",
            "--noplot",
        ],
        &[("CRITERION_HOME", &custom_root)],
    )
    .expect("helper binary must run");

    assert_eq!(
        exit_code(&output),
        Some(0),
        "{}",
        signal_description(&output)
    );
    let manifest = read_manifest(&manifest_path).expect("manifest must be readable");
    assert_eq!(manifest.output_root, custom_root);
    assert_eq!(manifest.cases.len(), 1);
    assert_eq!(manifest.bench_arguments, vec!["--noplot".to_owned()]);
    assert_eq!(manifest.tool_versions.criterion, "0.5.1");
}

/// One expected classification outcome.
#[derive(Clone, Copy, Debug)]
struct ClassifyCase {
    /// Lower bound of the mean change interval.
    lower: f64,
    /// Upper bound of the mean change interval.
    upper: f64,
    /// Expected process exit code.
    code: i32,
    /// Expected outcome name.
    outcome: &'static str,
}

#[rstest]
#[case::regression(ClassifyCase { lower: 0.30, upper: 0.45, code: 1, outcome: "regression" })]
#[case::within_noise(ClassifyCase { lower: -0.01, upper: 0.01, code: 0, outcome: "pass" })]
fn classify_reports_the_expected_outcome(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
    #[case] case: ClassifyCase,
) {
    let ClassifyCase {
        lower,
        upper,
        code: expected_code,
        outcome: expected_outcome,
    } = case;
    let workspace = sandbox.expect("sandbox must be created");
    Sandbox::write_baseline(&workspace.current_root, "fixture/alpha", "ci-current")
        .expect("current tree must be written");
    workspace
        .write_change("fixture/alpha", lower, upper)
        .expect("change estimates must be written");
    let preflight = workspace
        .write_preflight()
        .expect("preflight report must be written");
    let verdict_path = workspace.evidence.join("verdict.json");
    let github_output = workspace.evidence.join("github-output.txt");
    let step_summary = workspace.evidence.join("step-summary.md");
    fs::write(&github_output, "").expect("output file must be created");
    fs::write(&step_summary, "").expect("summary file must be created");

    let output = run(
        &binary,
        &[
            "classify",
            "--bench",
            "hnsw",
            "--preflight",
            &preflight.display().to_string(),
            "--verdict",
            &verdict_path.display().to_string(),
            "--threshold",
            "0.05",
        ],
        &[
            ("GITHUB_OUTPUT", &github_output),
            ("GITHUB_STEP_SUMMARY", &step_summary),
        ],
    )
    .expect("helper binary must run");

    assert_eq!(
        exit_code(&output),
        Some(expected_code),
        "{}",
        signal_description(&output)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(&format!("outcome={expected_outcome}")),
        "{stdout}"
    );
    let outputs = read(&github_output).expect("output file must be readable");
    assert!(
        outputs.contains(&format!("outcome={expected_outcome}")),
        "{outputs}"
    );
    let summary = read(&step_summary).expect("summary file must be readable");
    assert!(
        summary.contains(&format!("| Outcome | {expected_outcome} |")),
        "{summary}"
    );
    let verdict = BaselineVerdict::read(&verdict_path).expect("verdict must be readable");
    assert_eq!(verdict.outcome.as_str(), expected_outcome);
    assert_eq!(verdict.threshold, 0.05);
}

#[rstest]
fn no_reference_exits_four_without_recording_a_pass(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let workspace = sandbox.expect("sandbox must be created");
    let verdict_path = workspace.evidence.join("verdict.json");
    let github_output = workspace.evidence.join("github-output.txt");
    fs::write(&github_output, "").expect("output file must be created");

    let output = run(
        &binary,
        &[
            "classify",
            "--bench",
            "hnsw",
            "--preflight",
            &workspace.evidence.join("unused.json").display().to_string(),
            "--verdict",
            &verdict_path.display().to_string(),
            "--no-reference",
            "shallow",
        ],
        &[("GITHUB_OUTPUT", &github_output)],
    )
    .expect("helper binary must run");

    assert_eq!(
        exit_code(&output),
        Some(4),
        "{}",
        signal_description(&output)
    );
    let verdict = BaselineVerdict::read(&verdict_path).expect("verdict must be readable");
    assert_eq!(verdict.outcome, Outcome::NoReference);
    assert!(!verdict.to_line().contains("outcome=pass"));
    let outputs = read(&github_output).expect("output file must be readable");
    assert!(outputs.contains("outcome=no_reference"), "{outputs}");
}

#[rstest]
fn classify_rejects_an_unreadable_preflight_report(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let workspace = sandbox.expect("sandbox must be created");
    let bad = workspace.evidence.join("bad-preflight.json");
    fs::write(&bad, "{not json").expect("malformed report must be written");

    let output = run(
        &binary,
        &[
            "classify",
            "--bench",
            "hnsw",
            "--preflight",
            &bad.display().to_string(),
            "--verdict",
            &workspace
                .evidence
                .join("verdict.json")
                .display()
                .to_string(),
        ],
        &[],
    )
    .expect("helper binary must run");

    assert_eq!(
        exit_code(&output),
        Some(2),
        "{}",
        signal_description(&output)
    );
    assert!(
        stderr(&output).contains("bad-preflight.json"),
        "{}",
        stderr(&output)
    );
}
