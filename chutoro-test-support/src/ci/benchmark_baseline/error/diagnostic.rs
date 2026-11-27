//! Serializable diagnostics recorded in preflight reports and verdicts.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::super::Outcome;

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
        self.stage.clone_from(&stage.to_owned());
        self
    }
}
