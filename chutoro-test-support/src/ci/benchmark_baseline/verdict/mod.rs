//! The verdict record a benchmark comparison leaves behind.
//!
//! A verdict is the lane's durable evidence: the outcome, the revisions, the
//! threshold, the stage that failed if any, and one entry per benchmark
//! identifier. Its markdown rendering is what appears in the GitHub step
//! summary, so a reader can tell a regression from a broken lane without
//! opening a log.

use serde::{Deserialize, Serialize};

use super::manifest::read_json;
use super::manifest::write_json;
use super::{BaselineDiagnostic, BaselineError, Outcome, SCHEMA_VERSION};

mod case;
mod presentation;

pub use case::CaseVerdict;
pub use presentation::class_counts;

/// Verdict class assigned to one benchmark identifier.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseClass {
    /// The interval stayed within the noise threshold.
    WithinNoise,
    /// The interval lay wholly above the regression threshold.
    Regressed,
    /// The interval lay wholly below the negative threshold.
    Improved,
    /// The identifier had no reference data and was not compared.
    NewUnverified,
    /// The identifier had reference data and no current data.
    Removed,
    /// The identifier was compared but its evidence was unusable.
    Unreadable,
}

impl CaseClass {
    /// Return the stable lowercase name written to reports and summaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::CaseClass;
    ///
    /// assert_eq!(CaseClass::Regressed.as_str(), "regressed");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WithinNoise => "within_noise",
            Self::Regressed => "regressed",
            Self::Improved => "improved",
            Self::NewUnverified => "new_unverified",
            Self::Removed => "removed",
            Self::Unreadable => "unreadable",
        }
    }
}

/// Inputs for a verdict that records an outcome without comparing.
#[derive(Clone, Copy, Debug)]
pub struct NoComparison<'a> {
    /// Benchmark target.
    pub bench: &'a str,
    /// Outcome the lane recorded.
    pub outcome: Outcome,
    /// Stage that recorded it.
    pub stage: &'a str,
    /// Human-readable reason.
    pub reason: &'a str,
    /// Regression threshold the lane would have applied.
    pub threshold: f64,
}

/// Durable record of one benchmark baseline comparison.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BaselineVerdict {
    /// Schema version of this record.
    pub schema_version: u32,
    /// Benchmark target.
    pub bench: String,
    /// Outcome of the lane.
    pub outcome: Outcome,
    /// Stage that produced this verdict.
    pub stage: String,
    /// Stage that failed, when one did.
    pub failed_stage: Option<String>,
    /// Human-readable reason for a non-passing outcome.
    pub reason: Option<String>,
    /// Reference revision, when one existed.
    pub reference_revision: Option<String>,
    /// Current revision.
    pub current_revision: Option<String>,
    /// Regression threshold applied to the interval's lower bound.
    pub threshold: f64,
    /// Number of identifiers that could be compared.
    pub comparable_count: usize,
    /// Evidence for each identifier, sorted by identifier.
    pub cases: Vec<CaseVerdict>,
    /// Actionable diagnostics.
    pub diagnostics: Vec<BaselineDiagnostic>,
}

impl BaselineVerdict {
    /// Start a verdict for a lane that has only just begun.
    ///
    ///
    /// # Parameters
    ///
    /// - `bench`: benchmark target.
    /// - `threshold`: fractional regression threshold.
    ///
    /// # Returns
    ///
    /// A verdict stage `started` that no gate treats as a pass.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::BaselineVerdict;
    ///
    /// let verdict = BaselineVerdict::started("hnsw", 0.05);
    /// assert_eq!(verdict.stage, "started");
    /// ```
    #[must_use]
    pub fn started(bench: &str, threshold: f64) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            bench: bench.to_owned(),
            outcome: Outcome::Incomplete,
            stage: "started".to_owned(),
            failed_stage: None,
            reason: Some("the lane has not reached classification".to_owned()),
            reference_revision: None,
            current_revision: None,
            threshold,
            comparable_count: 0,
            cases: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Build a verdict for an outcome that needed no comparison.
    ///
    ///
    /// # Parameters
    ///
    /// - `request`: the outcome, stage, reason, and threshold to record.
    ///
    /// # Returns
    ///
    /// A verdict for an outcome that needed no comparison.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::Outcome;
    /// use chutoro_test_support::ci::benchmark_baseline::verdict::{
    ///     BaselineVerdict, NoComparison,
    /// };
    ///
    /// let verdict = BaselineVerdict::without_comparison(&NoComparison {
    ///     bench: "hnsw",
    ///     outcome: Outcome::NoReference,
    ///     stage: "classify",
    ///     reason: "shallow checkout",
    ///     threshold: 0.05,
    /// });
    /// assert_eq!(verdict.outcome, Outcome::NoReference);
    /// ```
    #[must_use]
    pub fn without_comparison(request: &NoComparison<'_>) -> Self {
        let diagnostic = BaselineDiagnostic {
            stage: request.stage.to_owned(),
            outcome: request.outcome,
            id: None,
            path: None,
            expected: None,
            remedy: "fetch enough history to compare against the previous commit".to_owned(),
            message: request.reason.to_owned(),
        };
        Self {
            schema_version: SCHEMA_VERSION,
            bench: request.bench.to_owned(),
            outcome: request.outcome,
            stage: request.stage.to_owned(),
            failed_stage: None,
            reason: Some(request.reason.to_owned()),
            reference_revision: None,
            current_revision: None,
            threshold: request.threshold,
            comparable_count: 0,
            cases: Vec::new(),
            diagnostics: vec![diagnostic],
        }
    }

    /// Record a failure and the stage that produced it.
    ///
    ///
    /// # Parameters
    ///
    /// - `stage`: stage that failed.
    /// - `error`: the failure to record.
    ///
    /// # Returns
    ///
    /// Nothing; the verdict is updated in place.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{BaselineError, BaselineVerdict};
    ///
    /// let error = BaselineError::NoReference {
    ///     reason: "shallow",
    ///     remedy: "fetch two commits".into(),
    /// };
    /// let mut verdict = BaselineVerdict::started("hnsw", 0.05);
    /// verdict.fail("stage", &error);
    /// assert_eq!(verdict.failed_stage.as_deref(), Some("stage"));
    /// ```
    pub fn fail(&mut self, stage: &str, error: &BaselineError) {
        self.outcome = error.outcome().unwrap_or(Outcome::Incomplete);
        self.stage.clone_from(&stage.to_owned());
        self.failed_stage = Some(stage.to_owned());
        self.reason = Some(error.to_string());
        self.diagnostics.push(error.diagnostic().with_stage(stage));
    }

    /// Return the identifiers classified as regressions.
    ///
    ///
    /// # Returns
    ///
    /// The identifiers classified as regressions.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{
    ///     BaselineVerdict, CaseClass, CaseVerdict,
    /// };
    ///
    /// let mut verdict = BaselineVerdict::started("hnsw", 0.05);
    /// verdict.cases.push(CaseVerdict::classified(
    ///     "fixture/alpha",
    ///     CaseClass::Regressed,
    ///     0.30,
    ///     0.40,
    /// ));
    /// assert_eq!(verdict.regressed_ids(), vec!["fixture/alpha"]);
    /// ```
    #[must_use]
    pub fn regressed_ids(&self) -> Vec<&str> {
        self.cases
            .iter()
            .filter(|case| case.class == CaseClass::Regressed)
            .map(|case| case.id.as_str())
            .collect()
    }

    /// Return the identifiers this lane could not compare.
    ///
    ///
    /// # Returns
    ///
    /// The identifiers the lane could not compare.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{BaselineVerdict, CaseClass, CaseVerdict};
    ///
    /// let mut verdict = BaselineVerdict::started("hnsw", 0.05);
    /// verdict.cases.push(CaseVerdict::not_compared(
    ///     "fixture/beta",
    ///     CaseClass::NewUnverified,
    ///     "no reference",
    /// ));
    /// assert_eq!(verdict.unverified_ids(), vec!["fixture/beta"]);
    /// ```
    #[must_use]
    pub fn unverified_ids(&self) -> Vec<&str> {
        self.cases
            .iter()
            .filter(|case| {
                matches!(
                    case.class,
                    CaseClass::NewUnverified | CaseClass::Removed | CaseClass::Unreadable
                )
            })
            .map(|case| case.id.as_str())
            .collect()
    }

    /// Write the verdict as pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when the verdict cannot be read, parsed, or written.
    ///
    /// # Examples
    ///
    /// ```
    /// # let directory = tempfile::tempdir().expect("temporary directory");
    /// use chutoro_test_support::ci::benchmark_baseline::BaselineVerdict;
    ///
    /// let path = directory.path().join("verdict.json");
    /// BaselineVerdict::started("hnsw", 0.05)
    ///     .write(&path)
    ///     .expect("verdict must be written");
    /// assert!(path.is_file());
    /// ```
    pub fn write(&self, path: &std::path::Path) -> Result<(), BaselineError> {
        write_json(path, self)
    }

    /// Read a verdict, naming the path when it cannot be parsed.
    ///
    /// # Errors
    ///
    /// Returns an error when the verdict cannot be read or parsed.
    pub fn read(path: &std::path::Path) -> Result<Self, BaselineError> {
        read_json(path)
    }
}
