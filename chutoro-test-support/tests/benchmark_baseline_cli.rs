//! Behavioural tests for the benchmark baseline helper binary.
//!
//! Each case builds a synthetic Criterion tree, runs the real binary with a
//! controlled environment, and asserts the documented exit code together with
//! the diagnostic it printed and the evidence it wrote. The exit-code mapping
//! is the contract the workflow relies on to distinguish a regression from a
//! lane defect.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use chutoro_test_support::ci::benchmark_baseline::preflight::{
    ComparableCase, ManifestSummary, PreflightReport,
};
use chutoro_test_support::ci::benchmark_baseline::{
    BaselineCase, BaselineManifest, BaselineVerdict, Outcome, RequiredFiles, ToolVersions,
};
use rstest::{fixture, rstest};
use tempfile::TempDir;

/// The helper binary Cargo built for this test target.
#[fixture]
fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_benchmark_baseline"))
}

/// Synthetic reference, current, and evidence directories.
struct Sandbox {
    /// Temporary root owning every path below.
    _root: TempDir,
    /// Current Criterion output root.
    current_root: PathBuf,
    /// Evidence directory.
    evidence: PathBuf,
}

impl Sandbox {
    /// Create the sandbox's directories.
    fn new() -> std::io::Result<Self> {
        let root = TempDir::new()?;
        let current_root = root.path().join("current-criterion");
        let evidence = root.path().join("evidence");
        fs::create_dir_all(&current_root)?;
        fs::create_dir_all(&evidence)?;
        Ok(Self {
            _root: root,
            current_root,
            evidence,
        })
    }

    /// Write a complete baseline tree for one identifier.
    fn write_baseline(&self, root: &Path, directory: &str, baseline: &str) -> std::io::Result<()> {
        let path = root.join(directory).join(baseline);
        fs::create_dir_all(&path)?;
        for file in ["benchmark.json", "estimates.json", "sample.json"] {
            fs::write(path.join(file), "{}")?;
        }
        Ok(())
    }

    /// Write a manifest for one side.
    fn write_manifest(&self, name: &str, manifest: &BaselineManifest) -> std::io::Result<PathBuf> {
        let path = self.evidence.join(name);
        let json = serde_json::to_vec_pretty(manifest).map_err(std::io::Error::other)?;
        fs::write(&path, json)?;
        Ok(path)
    }

    /// Write change estimates with the supplied mean interval.
    fn write_change(&self, directory: &str, lower: f64, upper: f64) -> std::io::Result<()> {
        let path = self.current_root.join(directory).join("change");
        fs::create_dir_all(&path)?;
        fs::write(
            path.join("estimates.json"),
            format!(
                "{{\"mean\":{{\"confidence_interval\":{{\"confidence_level\":0.95,\
                 \"lower_bound\":{lower},\"upper_bound\":{upper}}},\
                 \"point_estimate\":{lower},\"standard_error\":0.01}}}}"
            ),
        )
    }

    /// Write a preflight report naming one comparable identifier.
    fn write_preflight(&self) -> std::io::Result<PathBuf> {
        let current_summary = ManifestSummary {
            revision: "current".to_owned(),
            bench: "hnsw".to_owned(),
            baseline: "ci-current".to_owned(),
            output_root: self.current_root.clone(),
            criterion: "0.5.1".to_owned(),
            rustc: "rustc 1.99.1".to_owned(),
            bench_arguments: vec!["--noplot".to_owned()],
            environment: BTreeMap::new(),
        };
        let mut report = PreflightReport::new("stage", "hnsw");
        report.outcome = Outcome::Pass;
        report.reference = None;
        report.current = Some(current_summary);
        report.comparable = vec![ComparableCase {
            full_id: "fixture/alpha".to_owned(),
            reference_directory: "fixture/alpha".into(),
            current_directory: "fixture/alpha".into(),
        }];
        let path = self.evidence.join("preflight.json");
        report
            .write(&path)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(path)
    }
}

/// Build a sandbox or fail the test with the setup error.
#[fixture]
fn sandbox() -> Result<Sandbox, std::io::Error> {
    Sandbox::new()
}

/// Build a manifest with one complete case.
fn manifest(
    root: &Path,
    baseline: &str,
    criterion: &str,
    required: RequiredFiles,
) -> BaselineManifest {
    BaselineManifest {
        schema_version: 1,
        revision: format!("{baseline}-revision"),
        bench: "hnsw".to_owned(),
        baseline: baseline.to_owned(),
        output_root: root.to_path_buf(),
        tool_versions: ToolVersions {
            criterion: criterion.to_owned(),
            cargo: "cargo 1.99.1".to_owned(),
            rustc: "rustc 1.99.1".to_owned(),
        },
        bench_arguments: vec!["--noplot".to_owned()],
        environment: BTreeMap::new(),
        cases: vec![BaselineCase {
            full_id: "fixture/alpha".to_owned(),
            directory_name: "fixture/alpha".to_owned(),
            relative_directory: "fixture/alpha".into(),
            required_files: required,
        }],
    }
}

/// A complete required-file set.
fn complete() -> RequiredFiles {
    RequiredFiles {
        benchmark: true,
        estimates: true,
        sample: true,
    }
}

/// Run the helper binary with the supplied arguments and environment.
fn run(binary: &Path, arguments: &[&str], environment: &[(&str, &Path)]) -> Output {
    Command::new(binary)
        .args(arguments)
        .envs(environment.iter().copied())
        .output()
        .expect("helper binary must run")
}

/// Return the process exit code, failing when the process was signalled.
fn exit_code(output: &Output) -> i32 {
    output.status.code().unwrap_or_else(|| {
        panic!(
            "helper was signalled\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
    })
}

/// Return the helper's stderr as text.
fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Return an environment file's text.
fn read(path: &Path) -> String {
    fs::read_to_string(path).expect("environment file must be readable")
}

#[test]
fn usage_error_exits_sixty_four(binary: PathBuf) {
    let output = run(&binary, &[], &[]);

    assert_eq!(exit_code(&output), 64);
    assert!(
        stderr(&output).contains("no subcommand"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn help_exits_zero(binary: PathBuf) {
    let output = run(&binary, &["--help"], &[]);

    assert_eq!(exit_code(&output), 0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("inventory"), "{stdout}");
    assert!(stdout.contains("no_reference"), "{stdout}");
}

#[test]
fn missing_reference_sample_exits_incomplete_and_names_the_path(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let sandbox = sandbox.expect("sandbox must be created");
    let reference_root = sandbox.evidence.join("reference-criterion");
    sandbox
        .write_baseline(&reference_root, "fixture/alpha", "ci-reference")
        .expect("reference tree must be written");
    sandbox
        .write_baseline(&sandbox.current_root, "fixture/alpha", "ci-current")
        .expect("current tree must be written");
    let reference = manifest(
        &reference_root,
        "ci-reference",
        "0.5.1",
        RequiredFiles {
            sample: false,
            ..complete()
        },
    );
    let current = manifest(&sandbox.current_root, "ci-current", "0.5.1", complete());
    let reference_path = sandbox
        .write_manifest("reference-manifest.json", &reference)
        .expect("reference manifest must be written");
    let current_path = sandbox
        .write_manifest("current-manifest.json", &current)
        .expect("current manifest must be written");

    let output = run(
        &binary,
        &[
            "stage",
            "--bench",
            "hnsw",
            "--reference-manifest",
            &reference_path.display().to_string(),
            "--current-manifest",
            &current_path.display().to_string(),
            "--evidence",
            &sandbox.evidence.display().to_string(),
        ],
        &[],
    );

    assert_eq!(exit_code(&output), 2);
    let message = stderr(&output);
    assert!(message.contains("fixture/alpha"), "{message}");
    assert!(message.contains("sample.json"), "{message}");
    assert!(message.contains("ci-reference"), "{message}");
}

#[test]
fn version_mismatch_exits_incompatible_and_names_both_versions(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let sandbox = sandbox.expect("sandbox must be created");
    let reference_root = sandbox.evidence.join("reference-criterion");
    sandbox
        .write_baseline(&reference_root, "fixture/alpha", "ci-reference")
        .expect("reference tree must be written");
    sandbox
        .write_baseline(&sandbox.current_root, "fixture/alpha", "ci-current")
        .expect("current tree must be written");
    let reference = manifest(&reference_root, "ci-reference", "0.4.0", complete());
    let current = manifest(&sandbox.current_root, "ci-current", "0.5.1", complete());
    let reference_path = sandbox
        .write_manifest("reference-manifest.json", &reference)
        .expect("reference manifest must be written");
    let current_path = sandbox
        .write_manifest("current-manifest.json", &current)
        .expect("current manifest must be written");

    let output = run(
        &binary,
        &[
            "stage",
            "--bench",
            "hnsw",
            "--reference-manifest",
            &reference_path.display().to_string(),
            "--current-manifest",
            &current_path.display().to_string(),
            "--evidence",
            &sandbox.evidence.display().to_string(),
        ],
        &[],
    );

    assert_eq!(exit_code(&output), 3);
    let message = stderr(&output);
    assert!(message.contains("0.4.0"), "{message}");
    assert!(message.contains("0.5.1"), "{message}");
}

#[test]
fn inventory_honours_a_non_default_criterion_home(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let sandbox = sandbox.expect("sandbox must be created");
    let custom_root = sandbox.evidence.join("custom-criterion");
    sandbox
        .write_baseline(&custom_root, "fixture/alpha", "ci-reference")
        .expect("custom criterion tree must be written");
    let manifest_path = sandbox.evidence.join("reference-manifest.json");
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"));

    let output = run(
        &binary,
        &[
            "inventory",
            "--workspace",
            &workspace.display().to_string(),
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
    );

    assert_eq!(exit_code(&output), 0, "{}", stderr(&output));
    let manifest = BaselineManifest::read(&manifest_path).expect("manifest must be readable");
    assert_eq!(manifest.output_root, custom_root);
    assert_eq!(manifest.cases.len(), 1);
    assert_eq!(manifest.bench_arguments, vec!["--noplot".to_owned()]);
    assert_eq!(manifest.tool_versions.criterion, "0.5.1");
}

#[rstest]
#[case::regression(0.30, 0.45, 1, "regression")]
#[case::within_noise(-0.01, 0.01, 0, "pass")]
fn classify_reports_the_expected_outcome(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
    #[case] lower: f64,
    #[case] upper: f64,
    #[case] expected_code: i32,
    #[case] expected_outcome: &str,
) {
    let sandbox = sandbox.expect("sandbox must be created");
    sandbox
        .write_baseline(&sandbox.current_root, "fixture/alpha", "ci-current")
        .expect("current tree must be written");
    sandbox
        .write_change("fixture/alpha", lower, upper)
        .expect("change estimates must be written");
    let preflight = sandbox
        .write_preflight()
        .expect("preflight report must be written");
    let verdict_path = sandbox.evidence.join("verdict.json");
    let github_output = sandbox.evidence.join("github-output.txt");
    let step_summary = sandbox.evidence.join("step-summary.md");
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
    );

    assert_eq!(exit_code(&output), expected_code, "{}", stderr(&output));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(&format!("outcome={expected_outcome}")),
        "{stdout}"
    );
    let outputs = read(&github_output);
    assert!(
        outputs.contains(&format!("outcome={expected_outcome}")),
        "{outputs}"
    );
    let summary = read(&step_summary);
    assert!(
        summary.contains(&format!("| Outcome | {expected_outcome} |")),
        "{summary}"
    );
    let verdict = BaselineVerdict::read(&verdict_path).expect("verdict must be readable");
    assert_eq!(verdict.outcome.as_str(), expected_outcome);
    assert_eq!(verdict.threshold, 0.05);
}

#[test]
fn no_reference_exits_four_without_recording_a_pass(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let sandbox = sandbox.expect("sandbox must be created");
    let verdict_path = sandbox.evidence.join("verdict.json");
    let github_output = sandbox.evidence.join("github-output.txt");
    fs::write(&github_output, "").expect("output file must be created");

    let output = run(
        &binary,
        &[
            "classify",
            "--bench",
            "hnsw",
            "--preflight",
            &sandbox.evidence.join("unused.json").display().to_string(),
            "--verdict",
            &verdict_path.display().to_string(),
            "--no-reference",
            "shallow",
        ],
        &[("GITHUB_OUTPUT", &github_output)],
    );

    assert_eq!(exit_code(&output), 4);
    let verdict = BaselineVerdict::read(&verdict_path).expect("verdict must be readable");
    assert_eq!(verdict.outcome, Outcome::NoReference);
    assert!(!verdict.to_line().contains("outcome=pass"));
    let outputs = read(&github_output);
    assert!(outputs.contains("outcome=no_reference"), "{outputs}");
}

#[test]
fn classify_rejects_an_unreadable_preflight_report(
    binary: PathBuf,
    sandbox: Result<Sandbox, std::io::Error>,
) {
    let sandbox = sandbox.expect("sandbox must be created");
    let bad = sandbox.evidence.join("bad-preflight.json");
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
            &sandbox.evidence.join("verdict.json").display().to_string(),
        ],
        &[],
    );

    assert_eq!(exit_code(&output), 2);
    assert!(
        stderr(&output).contains("bad-preflight.json"),
        "{}",
        stderr(&output)
    );
}
