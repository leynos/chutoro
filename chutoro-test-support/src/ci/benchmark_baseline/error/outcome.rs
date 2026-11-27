//! Outcome and exit-code projection for [`BaselineError`](super::BaselineError).

use std::path::PathBuf;

use super::{BaselineDiagnostic, BaselineError};
use crate::ci::benchmark_baseline::{EXIT_USAGE, Outcome, exit_code};

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
    pub const fn outcome(&self) -> Option<Outcome> {
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
            | Self::InvertedInterval { .. }
            | Self::NonFiniteInterval { .. }
            | Self::InvalidThreshold { .. } => Some(Outcome::Incomplete),
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
        self.outcome().map_or(EXIT_USAGE, exit_code)
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
            | Self::StaleBaseline { path, .. } => (None, Some(path.clone()), None),

            Self::MetadataCommand { workspace, .. }
            | Self::MetadataMissingTarget { workspace, .. }
            | Self::MetadataJson { workspace, .. } => (None, Some(workspace.clone()), None),
            Self::DuplicateId { id, first, .. } => (Some(id.clone()), Some(first.clone()), None),
            Self::MissingRequiredFile { id, path, .. }
            | Self::MissingChange { id, path, .. }
            | Self::MalformedChange { id, path, .. } => {
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
            Self::ConfidenceLevel { id, expected, .. } => {
                (Some(id.clone()), None, Some(expected.to_string()))
            }
            Self::InvertedInterval { id, .. } | Self::NonFiniteInterval { id, .. } => {
                (Some(id.clone()), None, None)
            }
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
            | Self::Usage { .. }
            | Self::InvalidThreshold { .. } => (None, None, None),
        }
    }

    /// Return the remedy attached to this failure.
    const fn remedy(&self) -> &'static str {
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
            | Self::NonFiniteInterval { remedy, .. }
            | Self::InvalidThreshold { remedy, .. }
            | Self::NoReference { remedy, .. }
            | Self::Usage { remedy, .. } => remedy,
        }
    }
}
