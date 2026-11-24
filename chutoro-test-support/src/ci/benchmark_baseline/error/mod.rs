//! Typed failures for benchmark baseline discovery, staging, and comparison.
//!
//! Every variant names the stage, benchmark identifier, path, or tool version
//! that made the lane unable to produce comparable evidence, and the remedy a
//! maintainer should apply. `BaselineDiagnostic` is the serializable form used
//! in preflight reports and verdicts, and lives in `diagnostic`.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

pub mod diagnostic;
mod outcome;

pub use diagnostic::BaselineDiagnostic;

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
    /// The mean change interval was not finite.
    #[error("interval for benchmark id {id} is not finite; {remedy}")]
    NonFiniteInterval {
        /// Benchmark identifier.
        id: String,
        /// Remedy for the maintainer.
        remedy: &'static str,
    },
    /// The regression threshold is not a positive fraction.
    #[error("regression threshold {threshold} is not a finite positive fraction; {remedy}")]
    InvalidThreshold {
        /// Threshold as supplied.
        threshold: f64,
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
