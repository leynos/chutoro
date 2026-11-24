//! Typed failures for benchmark baseline discovery, staging, and comparison.
//!
//! Every variant names the stage, benchmark identifier, path, or tool version
//! that made the lane unable to produce comparable evidence, and the remedy a
//! maintainer should apply. `BaselineDiagnostic` is the serializable form used
//! in preflight reports and verdicts.

use std::io;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::Outcome;

/// One actionable diagnostic recorded in a preflight report or verdict.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BaselineDiagnostic {
    /// Pipeline stage that produced the diagnostic.
    pub stage: String,
    /// Outcome the diagnostic contributes to.
    pub outcome: Outcome,
    /// Benchmark identifier, when the failure concerns one identifier.
    pub id: Option<String>,
    /// Path the diagnostic concerns, when it concerns one path.
    pub path: Option<PathBuf>,
    /// What the helper expected to find.
    pub expected: Option<String>,
    /// Remedy a maintainer should apply.
    pub remedy: String,
    /// Human-readable description of the failure.
    pub message: String,
}

impl BaselineDiagnostic {
    /// Attach the stage that was running when the diagnostic was produced.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{
    ///     BaselineDiagnostic, Outcome,
    /// };
    ///
    /// let diagnostic = BaselineDiagnostic {
    ///     stage: "unset".to_owned(),
    ///     outcome: Outcome::Incomplete,
    ///     id: None,
    ///     path: None,
    ///     expected: None,
    ///     remedy: "rerun the lane".to_owned(),
    ///     message: "example".to_owned(),
    /// };
    /// assert_eq!(diagnostic.with_stage("stage").stage, "stage");
    /// ```
    #[must_use]
    pub fn with_stage(mut self, stage: &str) -> Self {
        self.stage = stage.to_owned();
        self
    }
}

/// Failure raised while discovering, staging, or classifying baselines.
#[derive(Debug, Error)]
pub enum BaselineError {
    /// The Criterion output root could not be resolved.
    #[error("criterion output root cannot be resolved: {reason}; {remedy}")]
    OutputRoot {
        /// Why resolution failed.
        reason: String,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// Running `cargo metadata` failed.
    #[error("cargo metadata failed in {workspace}: {message}; {remedy}")]
    MetadataCommand {
        /// Directory in which metadata was requested.
        workspace: PathBuf,
        /// Captured diagnostic text.
        message: String,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// `cargo metadata` succeeded but omitted the target directory.
    #[error("cargo metadata in {workspace} reported no target directory; {remedy}")]
    MetadataMissingTarget {
        /// Directory in which metadata was requested.
        workspace: PathBuf,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// `cargo metadata` produced output that is not valid JSON.
    #[error("cargo metadata in {workspace} produced unreadable JSON: {source}; {remedy}")]
    MetadataJson {
        /// Directory in which metadata was requested.
        workspace: PathBuf,
        /// Underlying JSON failure.
        source: serde_json::Error,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A tool version could not be recorded.
    #[error("cannot record the {tool} version: {message}; {remedy}")]
    ToolVersion {
        /// Tool whose version was requested.
        tool: &'static str,
        /// Captured diagnostic text.
        message: String,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A file could not be read.
    #[error("cannot read {path}: {source}; {remedy}")]
    Read {
        /// Path that could not be read.
        path: PathBuf,
        /// Underlying I/O failure.
        source: io::Error,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A file could not be written.
    #[error("cannot write {path}: {source}; {remedy}")]
    Write {
        /// Path that could not be written.
        path: PathBuf,
        /// Underlying I/O failure.
        source: io::Error,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A manifest or report could not be parsed.
    #[error("cannot parse {path}: {source}; {remedy}")]
    Parse {
        /// Path that could not be parsed.
        path: PathBuf,
        /// Underlying JSON failure.
        source: serde_json::Error,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A manifest or report could not be serialized.
    #[error("cannot serialize {path}: {source}; {remedy}")]
    Serialize {
        /// Path that could not be written.
        path: PathBuf,
        /// Underlying JSON failure.
        source: serde_json::Error,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A manifest declared a schema this helper does not understand.
    #[error("unsupported manifest schema {found} in {path} (expected {expected}); {remedy}")]
    Schema {
        /// Path of the manifest.
        path: PathBuf,
        /// Schema version found.
        found: u32,
        /// Schema version expected.
        expected: u32,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// The same benchmark identifier appeared more than once.
    #[error("benchmark id {id} appears twice ({first} and {second}); {remedy}")]
    DuplicateId {
        /// Duplicated benchmark identifier.
        id: String,
        /// First recorded location.
        first: PathBuf,
        /// Second recorded location.
        second: PathBuf,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A required file was absent.
    #[error("benchmark id {id} is missing required file {path}; {remedy}")]
    MissingRequiredFile {
        /// Benchmark identifier.
        id: String,
        /// Path that was absent.
        path: PathBuf,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A stale directory could not be removed.
    #[error("cannot remove {}: {source}; {remedy}", path.display())]
    Remove {
        /// Path that could not be removed.
        path: PathBuf,
        /// Underlying I/O failure.
        source: io::Error,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A baseline file predates the freshness marker.
    #[error("baseline file {path} is older than {marker}; {remedy}")]
    StaleBaseline {
        /// Baseline file that is too old.
        path: PathBuf,
        /// Freshness marker file.
        marker: PathBuf,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// The two sides share no comparable benchmark identifiers.
    #[error(
        "no comparable benchmark ids for {bench}: reference baseline \
         {reference_baseline} and current baseline {current_baseline}; {remedy}"
    )]
    EmptyComparable {
        /// Benchmark target.
        bench: String,
        /// Reference baseline name.
        reference_baseline: String,
        /// Current baseline name.
        current_baseline: String,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A compatibility field differed between the two sides.
    #[error("incompatible {field}: reference {reference}, current {current}; {remedy}")]
    Incompatible {
        /// Field that differed.
        field: &'static str,
        /// Reference-side value.
        reference: String,
        /// Current-side value.
        current: String,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// Staging one benchmark identifier failed.
    #[error(
        "cannot stage benchmark id {id} from {source_directory} to {destination_directory}: \
         {message}; {remedy}"
    )]
    StageCopy {
        /// Benchmark identifier being staged.
        id: String,
        /// Source baseline directory.
        source_directory: PathBuf,
        /// Destination baseline directory.
        destination_directory: PathBuf,
        /// Underlying failure description.
        message: String,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// A comparison did not write change estimates for an identifier.
    #[error("benchmark id {id} has no change estimates at {path}; {remedy}")]
    MissingChange {
        /// Benchmark identifier.
        id: String,
        /// Expected change estimates path.
        path: PathBuf,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// Change estimates could not be parsed.
    #[error("benchmark id {id} has malformed change estimates at {path}: {source}; {remedy}")]
    MalformedChange {
        /// Benchmark identifier.
        id: String,
        /// Path of the malformed file.
        path: PathBuf,
        /// Underlying JSON failure.
        source: serde_json::Error,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// The comparison used a different confidence level.
    #[error("benchmark id {id} used confidence level {found}, expected {expected}; {remedy}")]
    ConfidenceLevel {
        /// Benchmark identifier.
        id: String,
        /// Confidence level found.
        found: f64,
        /// Confidence level required.
        expected: f64,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// The mean change interval was inverted.
    #[error("interval for benchmark id {id} is inverted ({lower} > {upper}); {remedy}")]
    InvertedInterval {
        /// Benchmark identifier.
        id: String,
        /// Lower bound.
        lower: f64,
        /// Upper bound.
        upper: f64,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// No previous revision existed to compare against.
    #[error("no reference commit ({reason}); {remedy}")]
    NoReference {
        /// Why no reference exists: `shallow` or `root-commit`.
        reason: &'static str,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// The helper was invoked with invalid arguments.
    #[error("invalid arguments: {message}; {remedy}")]
    Usage {
        /// Description of the invalid arguments.
        message: String,
        /// Usage remedy.
        remedy: &'static str,
    },
}

impl BaselineError {
    /// Return the outcome this failure contributes to a verdict.
    ///
    /// Usage errors return `None` because they never produce a verdict.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{
    ///     BaselineError, Outcome,
    /// };
    ///
    /// let error = BaselineError::Usage {
    ///     message: "missing subcommand".to_owned(),
    ///     remedy: "run with --help".into(),
    /// };
    /// assert_eq!(error.outcome(), None);
    /// ```
    #[must_use]
    pub fn outcome(&self) -> Option<Outcome> {
        match self {
            Self::Incompatible { .. } | Self::Schema { .. } | Self::ConfidenceLevel { .. } => {
                Some(Outcome::Incompatible)
            }
            Self::NoReference { .. } => Some(Outcome::NoReference),
            Self::Usage { .. } => None,
            Self::OutputRoot { .. }
            | Self::MetadataCommand { .. }
            | Self::MetadataMissingTarget { .. }
            | Self::MetadataJson { .. }
            | Self::ToolVersion { .. }
            | Self::Read { .. }
            | Self::Write { .. }
            | Self::Remove { .. }
            | Self::Parse { .. }
            | Self::Serialize { .. }
            | Self::DuplicateId { .. }
            | Self::MissingRequiredFile { .. }
            | Self::StaleBaseline { .. }
            | Self::EmptyComparable { .. }
            | Self::StageCopy { .. }
            | Self::MissingChange { .. }
            | Self::MalformedChange { .. }
            | Self::InvertedInterval { .. } => Some(Outcome::Incomplete),
        }
    }

    /// Return the helper exit code for this failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{
    ///     BaselineError, EXIT_USAGE,
    /// };
    ///
    /// let error = BaselineError::Usage {
    ///     message: "missing subcommand".to_owned(),
    ///     remedy: "run with --help".into(),
    /// };
    /// assert_eq!(error.exit_code(), EXIT_USAGE);
    /// ```
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        self.outcome().map_or(super::EXIT_USAGE, super::exit_code)
    }

    /// Convert the failure into a serializable diagnostic.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::BaselineError;
    ///
    /// let error = BaselineError::NoReference {
    ///     reason: "shallow",
    ///     remedy: "fetch two commits".into(),
    /// };
    /// assert_eq!(error.diagnostic().stage, "unset");
    /// ```
    #[must_use]
    pub fn diagnostic(&self) -> BaselineDiagnostic {
        let (id, path, expected) = self.diagnostic_location();
        BaselineDiagnostic {
            stage: "unset".to_owned(),
            outcome: self.outcome().unwrap_or(Outcome::Incomplete),
            id,
            path,
            expected,
            remedy: self.remedy().to_owned(),
            message: self.to_string(),
        }
    }

    /// Return the identifier, path, and expectation named by this failure.
    fn diagnostic_location(&self) -> (Option<String>, Option<PathBuf>, Option<String>) {
        match self {
            Self::Read { path, .. }
            | Self::Write { path, .. }
            | Self::Remove { path, .. }
            | Self::Parse { path, .. }
            | Self::Serialize { path, .. }
            | Self::Schema { path, .. }
            | Self::StaleBaseline { path, .. }
            | Self::MissingChange { path, .. } => (None, Some(path.clone()), None),
            Self::MetadataCommand { workspace, .. }
            | Self::MetadataMissingTarget { workspace, .. }
            | Self::MetadataJson { workspace, .. } => (None, Some(workspace.clone()), None),
            Self::DuplicateId { id, first, .. } => (Some(id.clone()), Some(first.clone()), None),
            Self::MissingRequiredFile { id, path, .. } => {
                (Some(id.clone()), Some(path.clone()), None)
            }
            Self::StageCopy {
                id,
                source_directory,
                destination_directory,
                ..
            } => (
                Some(id.clone()),
                Some(destination_directory.clone()),
                Some(source_directory.display().to_string()),
            ),
            Self::MalformedChange { id, path, .. } => (Some(id.clone()), Some(path.clone()), None),
            Self::ConfidenceLevel { id, expected, .. } => {
                (Some(id.clone()), None, Some(expected.to_string()))
            }
            Self::InvertedInterval { id, .. } => (Some(id.clone()), None, None),
            Self::EmptyComparable { bench, .. } => (
                Some(bench.clone()),
                None,
                Some("at least one shared benchmark id".to_owned()),
            ),
            Self::Incompatible {
                field, reference, ..
            } => (None, None, Some(format!("{field}={reference}"))),
            Self::OutputRoot { .. }
            | Self::ToolVersion { .. }
            | Self::NoReference { .. }
            | Self::Usage { .. } => (None, None, None),
        }
    }

    /// Return the remedy attached to this failure.
    fn remedy(&self) -> &'static str {
        match self {
            Self::OutputRoot { remedy, .. }
            | Self::MetadataCommand { remedy, .. }
            | Self::MetadataMissingTarget { remedy, .. }
            | Self::MetadataJson { remedy, .. }
            | Self::ToolVersion { remedy, .. }
            | Self::Read { remedy, .. }
            | Self::Write { remedy, .. }
            | Self::Remove { remedy, .. }
            | Self::Parse { remedy, .. }
            | Self::Serialize { remedy, .. }
            | Self::Schema { remedy, .. }
            | Self::DuplicateId { remedy, .. }
            | Self::MissingRequiredFile { remedy, .. }
            | Self::StaleBaseline { remedy, .. }
            | Self::EmptyComparable { remedy, .. }
            | Self::Incompatible { remedy, .. }
            | Self::StageCopy { remedy, .. }
            | Self::MissingChange { remedy, .. }
            | Self::MalformedChange { remedy, .. }
            | Self::ConfidenceLevel { remedy, .. }
            | Self::InvertedInterval { remedy, .. }
            | Self::NoReference { remedy, .. }
            | Self::Usage { remedy, .. } => remedy,
        }
    }
}
