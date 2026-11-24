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

use chutoro_test_support::ci::benchmark_baseline::manifest::read_manifest;
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
    ///
    /// The `benchmark.json` carries Criterion's own identifier shape, so a
    /// real inventory of this tree succeeds; the other two files only need to
    /// exist for the required-file checks.
    fn write_baseline(root: &Path, directory: &str, baseline: &str) -> std::io::Result<()> {
        let path = root.join(directory).join(baseline);
        fs::create_dir_all(&path)?;
        fs::write(
            path.join("benchmark.json"),
            format!(r#"{{"full_id":"{directory}","directory_name":"{directory}"}}"#),
        )?;
        for file in ["estimates.json", "sample.json"] {
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
const fn complete() -> RequiredFiles {
    RequiredFiles {
        benchmark: true,
        estimates: true,
        sample: true,
    }
}

/// Run the helper binary with the supplied arguments and environment.
#[expect(
    clippy::expect_used,
    reason = "a helper that cannot spawn the binary is a broken test harness"
)]
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
#[expect(
    clippy::expect_used,
    reason = "the evidence file must exist for the assertion to mean anything"
)]
fn read(path: &Path) -> String {
    fs::read_to_string(path).expect("environment file must be readable")
}

#[rstest]
fn usage_error_exits_sixty_four(binary: PathBuf) {
    let output = run(&binary, &[], &[]);

    assert_eq!(exit_code(&output), 64);
    assert!(
        stderr(&output).contains("no subcommand"),
        "{}",
        stderr(&output)
    );
}

#[rstest]
fn help_exits_zero(binary: PathBuf) {
    let output = run(&binary, &["--help"], &[]);

    assert_eq!(exit_code(&output), 0);
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
    let reference_root = workspace.evidence.join("reference-criterion");
    Sandbox::write_baseline(&reference_root, "fixture/alpha", "ci-reference")
        .expect("reference tree must be written");
    Sandbox::write_baseline(&workspace.current_root, "fixture/alpha", "ci-current")
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
    let current = manifest(&workspace.current_root, "ci-current", "0.5.1", complete());
    let reference_path = workspace
        .write_manifest("reference-manifest.json", &reference)
        .expect("reference manifest must be written");
    let current_path = workspace
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
            &workspace.evidence.display().to_string(),
        ],
        &[],
    );

    assert_eq!(exit_code(&output), 2);
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
    let reference_root = workspace.evidence.join("reference-criterion");
    Sandbox::write_baseline(&reference_root, "fixture/alpha", "ci-reference")
        .expect("reference tree must be written");
    Sandbox::write_baseline(&workspace.current_root, "fixture/alpha", "ci-current")
        .expect("current tree must be written");
    let reference = manifest(&reference_root, "ci-reference", "0.4.0", complete());
    let current = manifest(&workspace.current_root, "ci-current", "0.5.1", complete());
    let reference_path = workspace
        .write_manifest("reference-manifest.json", &reference)
        .expect("reference manifest must be written");
    let current_path = workspace
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
            &workspace.evidence.display().to_string(),
        ],
        &[],
    );

    assert_eq!(exit_code(&output), 3);
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
    );

    assert_eq!(exit_code(&output), 0, "{}", stderr(&output));
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
    );

    assert_eq!(exit_code(&output), 4);
    let verdict = BaselineVerdict::read(&verdict_path).expect("verdict must be readable");
    assert_eq!(verdict.outcome, Outcome::NoReference);
    assert!(!verdict.to_line().contains("outcome=pass"));
    let outputs = read(&github_output);
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
    );

    assert_eq!(exit_code(&output), 2);
    assert!(
        stderr(&output).contains("bad-preflight.json"),
        "{}",
        stderr(&output)
    );
}
