//! Harness shared by the benchmark baseline CLI tests.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use chutoro_test_support::ci::benchmark_baseline::preflight::{
    ComparableCase, ManifestSummary, PreflightReport,
};
use chutoro_test_support::ci::benchmark_baseline::{
    BaselineCase, BaselineManifest, Outcome, RequiredFiles, ToolVersions,
};
use rstest::fixture;
use tempfile::TempDir;

/// The helper binary Cargo built for this test target.
#[fixture]
pub fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_benchmark_baseline"))
}

/// Synthetic reference, current, and evidence directories.
pub struct Sandbox {
    /// Temporary root owning every path below.
    _root: TempDir,
    /// Current Criterion output root.
    pub current_root: PathBuf,
    /// Evidence directory.
    pub evidence: PathBuf,
}

impl Sandbox {
    /// Create the sandbox's directories.
    pub fn new() -> std::io::Result<Self> {
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
    pub fn write_baseline(root: &Path, directory: &str, baseline: &str) -> std::io::Result<()> {
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
    pub fn write_manifest(
        &self,
        name: &str,
        manifest: &BaselineManifest,
    ) -> std::io::Result<PathBuf> {
        let path = self.evidence.join(name);
        let json = serde_json::to_vec_pretty(manifest).map_err(std::io::Error::other)?;
        fs::write(&path, json)?;
        Ok(path)
    }

    /// Write change estimates with the supplied mean interval.
    pub fn write_change(&self, directory: &str, lower: f64, upper: f64) -> std::io::Result<()> {
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
    pub fn write_preflight(&self) -> std::io::Result<PathBuf> {
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
pub fn sandbox() -> Result<Sandbox, std::io::Error> {
    Sandbox::new()
}

/// Build a manifest with one complete case.
pub fn manifest(
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
pub const fn complete() -> RequiredFiles {
    RequiredFiles {
        benchmark: true,
        estimates: true,
        sample: true,
    }
}

/// Run the helper binary with the supplied arguments and environment.
pub fn run(
    binary: &Path,
    arguments: &[&str],
    environment: &[(&str, &Path)],
) -> std::io::Result<Output> {
    Command::new(binary)
        .args(arguments)
        // An inherited summary sink would receive a synthetic verdict's
        // Markdown; a test that wants one sets it explicitly.
        .env_remove("GITHUB_STEP_SUMMARY")
        .envs(environment.iter().copied())
        .output()
}

/// Stage a synthetic reference and current tree and run the helper's `stage`.
///
/// `reference` is the reference side's recorded Criterion version and
/// required-file set; `current_version` is the current side's version. The
/// current side always carries every required file.
pub fn run_stage(
    binary: &Path,
    workspace: &Sandbox,
    reference: (&str, RequiredFiles),
    current_version: &str,
) -> std::io::Result<Output> {
    let reference_root = workspace.evidence.join("reference-criterion");
    Sandbox::write_baseline(&reference_root, "fixture/alpha", "ci-reference")?;
    Sandbox::write_baseline(&workspace.current_root, "fixture/alpha", "ci-current")?;
    let (version, files) = reference;
    let reference_manifest = manifest(&reference_root, "ci-reference", version, files);
    let current_manifest = manifest(
        &workspace.current_root,
        "ci-current",
        current_version,
        complete(),
    );
    let reference_path =
        workspace.write_manifest("reference-manifest.json", &reference_manifest)?;
    let current_path = workspace.write_manifest("current-manifest.json", &current_manifest)?;

    run(
        binary,
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
    )
}

/// Return the process exit code, or `None` when the process was signalled.
///
/// Callers assert on the value, so a signalled helper fails the test with its
/// captured output rather than panicking inside this shared helper.
pub fn exit_code(output: &Output) -> Option<i32> {
    output.status.code()
}

/// Describe a signalled helper for a test's failure message.
pub fn signal_description(output: &Output) -> String {
    format!(
        "helper was signalled\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Return the helper's stderr as text.
pub fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Return an environment file's text.
pub fn read(path: &Path) -> std::io::Result<String> {
    fs::read_to_string(path)
}
