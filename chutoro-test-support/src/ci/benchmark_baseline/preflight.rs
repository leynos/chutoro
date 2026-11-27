//! Preflight evidence shared by the staging and classification stages.
//!
//! The staging stage writes one preflight report whether it succeeds or fails,
//! so the workflow always has a record of the resolved roots, the identifiers
//! it could compare, and the diagnosis when it could not. The classification
//! stage reads that report instead of re-deriving the layout.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::manifest::{BaselineManifest, read_json, write_json};
use super::{BaselineDiagnostic, BaselineError, Outcome, SCHEMA_VERSION};

/// Comparable facts about one side of a comparison.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ManifestSummary {
    /// Revision the baseline was measured from.
    pub revision: String,
    /// Benchmark target.
    pub bench: String,
    /// Criterion baseline name.
    pub baseline: String,
    /// Resolved Criterion output root.
    pub output_root: PathBuf,
    /// Resolved Criterion version.
    pub criterion: String,
    /// `rustc -Vv` output.
    pub rustc: String,
    /// Measurement arguments recorded for compatibility.
    pub bench_arguments: Vec<String>,
    /// `CHUTORO_BENCH_*` values recorded for compatibility.
    pub environment: BTreeMap<String, String>,
}

impl ManifestSummary {
    /// Summarize a manifest for a preflight report.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # use std::path::Path;
    /// use chutoro_test_support::ci::benchmark_baseline::preflight::ManifestSummary;
    /// use chutoro_test_support::ci::benchmark_baseline::{BaselineManifest, ToolVersions};
    ///
    /// let manifest = BaselineManifest {
    ///     schema_version: 1,
    ///     revision: "abc".to_owned(),
    ///     bench: "hnsw".to_owned(),
    ///     baseline: "ci-current".to_owned(),
    ///     output_root: Path::new("/work/target/criterion").to_path_buf(),
    ///     tool_versions: ToolVersions {
    ///         criterion: "0.5.1".to_owned(),
    ///         cargo: "cargo 1.99.1".to_owned(),
    ///         rustc: "rustc 1.99.1".to_owned(),
    ///     },
    ///     bench_arguments: Vec::new(),
    ///     environment: BTreeMap::new(),
    ///     cases: Vec::new(),
    /// };
    /// let summary = ManifestSummary::from_manifest(&manifest);
    /// assert_eq!(summary.revision, "abc");
    /// ```
    #[must_use]
    pub fn from_manifest(manifest: &BaselineManifest) -> Self {
        Self {
            revision: manifest.revision.clone(),
            bench: manifest.bench.clone(),
            baseline: manifest.baseline.clone(),
            output_root: manifest.output_root.clone(),
            criterion: manifest.tool_versions.criterion.clone(),
            rustc: manifest.tool_versions.rustc.clone(),
            bench_arguments: manifest.bench_arguments.clone(),
            environment: manifest.environment.clone(),
        }
    }
}

/// One benchmark identifier present on both sides.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ComparableCase {
    /// Criterion's full benchmark identifier.
    pub full_id: String,
    /// Identifier directory relative to the reference output root.
    pub reference_directory: PathBuf,
    /// Identifier directory relative to the current output root.
    pub current_directory: PathBuf,
}

/// Evidence written by the staging stage and consumed by classification.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreflightReport {
    /// Schema version of this record.
    pub schema_version: u32,
    /// Stage that wrote the report.
    pub stage: String,
    /// Outcome the evidence supports.
    pub outcome: Outcome,
    /// Human-readable reason for a non-passing outcome.
    pub reason: Option<String>,
    /// Benchmark target.
    pub bench: String,
    /// Reference-side facts, when a manifest was readable.
    pub reference: Option<ManifestSummary>,
    /// Current-side facts, when a manifest was readable.
    pub current: Option<ManifestSummary>,
    /// Identifiers that can be compared.
    pub comparable: Vec<ComparableCase>,
    /// Identifiers present only on the current side.
    pub new_unverified: Vec<String>,
    /// Identifiers present only on the reference side.
    pub removed: Vec<String>,
    /// Actionable diagnostics.
    pub diagnostics: Vec<BaselineDiagnostic>,
}

impl PreflightReport {
    /// Start an incomplete report that a later stage fills in.
    ///
    /// The constructor deliberately starts at `incomplete`: a report becomes
    /// `Pass` only when staging has written every comparable baseline and the
    /// filter, so no caller can publish success by forgetting a step.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::preflight::PreflightReport;
    ///
    /// let report = PreflightReport::new("stage", "hnsw");
    /// assert_eq!(report.outcome, chutoro_test_support::ci::benchmark_baseline::Outcome::Incomplete);
    /// ```
    #[must_use]
    pub fn new(stage: &str, bench: &str) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            stage: stage.to_owned(),
            outcome: Outcome::Incomplete,
            reason: None,
            bench: bench.to_owned(),
            reference: None,
            current: None,
            comparable: Vec::new(),
            new_unverified: Vec::new(),
            removed: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Build a report from a failure caught after both manifests were read.
    #[must_use]
    pub fn failure(stage: &str, bench: &str, error: &BaselineError) -> Self {
        let outcome = error.outcome().unwrap_or(Outcome::Incomplete);
        let diagnostic = error.diagnostic().with_stage(stage);
        Self {
            schema_version: SCHEMA_VERSION,
            stage: stage.to_owned(),
            outcome,
            reason: Some(error.to_string()),
            bench: bench.to_owned(),
            reference: None,
            current: None,
            comparable: Vec::new(),
            new_unverified: Vec::new(),
            removed: Vec::new(),
            diagnostics: vec![diagnostic],
        }
    }

    /// Write the report as pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when the report cannot be read, parsed, or written.
    ///
    /// # Examples
    ///
    /// ```
    /// # let directory = tempfile::tempdir().expect("temporary directory");
    /// use chutoro_test_support::ci::benchmark_baseline::preflight::PreflightReport;
    ///
    /// let path = directory.path().join("preflight.json");
    /// PreflightReport::new("stage", "hnsw")
    ///     .write(&path)
    ///     .expect("report must be written");
    /// assert!(path.is_file());
    /// ```
    pub fn write(&self, path: &Path) -> Result<(), BaselineError> {
        write_json(path, self)
    }

    /// Read a report, naming the path when it cannot be parsed.
    ///
    /// # Errors
    ///
    /// Returns an error when the report cannot be read or parsed, or when it
    /// declares a schema version this helper does not understand.
    pub fn read(path: &Path) -> Result<Self, BaselineError> {
        let report: Self = read_json(path)?;
        if report.schema_version != SCHEMA_VERSION {
            return Err(BaselineError::Schema {
                path: path.to_path_buf(),
                found: report.schema_version,
                expected: SCHEMA_VERSION,
                remedy: "rerun staging with this helper version",
            });
        }

        Ok(report)
    }
}
