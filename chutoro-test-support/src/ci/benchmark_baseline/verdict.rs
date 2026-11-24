//! The verdict record a benchmark comparison leaves behind.
//!
//! A verdict is the lane's durable evidence: the outcome, the revisions, the
//! threshold, the stage that failed if any, and one entry per benchmark
//! identifier. Its markdown rendering is what appears in the GitHub step
//! summary, so a reader can tell a regression from a broken lane without
//! opening a log.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use super::manifest::read_json;
use super::manifest::write_json;
use super::{BaselineDiagnostic, BaselineError, Outcome, SCHEMA_VERSION};

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

/// Evidence for one benchmark identifier.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CaseVerdict {
    /// Criterion's full benchmark identifier.
    pub id: String,
    /// Verdict class assigned to the identifier.
    pub class: CaseClass,
    /// Lower bound of the mean change interval, when compared.
    pub lower_bound: Option<f64>,
    /// Upper bound of the mean change interval, when compared.
    pub upper_bound: Option<f64>,
    /// Reason the identifier could not be compared, when it could not.
    pub reason: Option<String>,
}

impl CaseVerdict {
    /// Build a verdict entry for a classified interval.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::CaseVerdict;
    /// use chutoro_test_support::ci::benchmark_baseline::CaseClass;
    ///
    /// let verdict = CaseVerdict::classified("fixture/alpha", CaseClass::WithinNoise, 0.01, 0.02);
    /// assert_eq!(verdict.class, CaseClass::WithinNoise);
    /// ```
    #[must_use]
    pub fn classified(id: &str, class: CaseClass, lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            id: id.to_owned(),
            class,
            lower_bound: Some(lower_bound),
            upper_bound: Some(upper_bound),
            reason: None,
        }
    }

    /// Build a verdict entry for an identifier that was not compared.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{CaseClass, CaseVerdict};
    ///
    /// let verdict = CaseVerdict::not_compared("fixture/beta", CaseClass::NewUnverified, "no reference");
    /// assert!(verdict.lower_bound.is_none());
    /// ```
    #[must_use]
    pub fn not_compared(id: &str, class: CaseClass, reason: &str) -> Self {
        Self {
            id: id.to_owned(),
            class,
            lower_bound: None,
            upper_bound: None,
            reason: Some(reason.to_owned()),
        }
    }
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
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{BaselineVerdict, Outcome};
    ///
    /// let verdict = BaselineVerdict::without_comparison(
    ///     "hnsw",
    ///     Outcome::NoReference,
    ///     "classify",
    ///     "shallow checkout",
    ///     0.05,
    /// );
    /// assert_eq!(verdict.outcome, Outcome::NoReference);
    /// ```
    #[must_use]
    pub fn without_comparison(
        bench: &str,
        outcome: Outcome,
        stage: &str,
        reason: &str,
        threshold: f64,
    ) -> Self {
        let diagnostic = BaselineDiagnostic {
            stage: stage.to_owned(),
            outcome,
            id: None,
            path: None,
            expected: None,
            remedy: "fetch enough history to compare against the previous commit".to_owned(),
            message: reason.to_owned(),
        };
        Self {
            schema_version: SCHEMA_VERSION,
            bench: bench.to_owned(),
            outcome,
            stage: stage.to_owned(),
            failed_stage: None,
            reason: Some(reason.to_owned()),
            reference_revision: None,
            current_revision: None,
            threshold,
            comparable_count: 0,
            cases: Vec::new(),
            diagnostics: vec![diagnostic],
        }
    }

    /// Record a failure and the stage that produced it.
    ///
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
        self.stage = stage.to_owned();
        self.failed_stage = Some(stage.to_owned());
        self.reason = Some(error.to_string());
        self.diagnostics.push(error.diagnostic().with_stage(stage));
    }

    /// Return the identifiers classified as regressions.
    ///
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
    pub fn read(path: &std::path::Path) -> Result<Self, BaselineError> {
        read_json(path)
    }

    /// Render the verdict as a Markdown table for the step summary.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::BaselineVerdict;
    ///
    /// let summary = BaselineVerdict::started("hnsw", 0.05).to_markdown();
    /// assert!(summary.contains("| Outcome | incomplete |"));
    /// ```
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut summary = String::new();
        let _write_result = writeln!(summary, "## Benchmark baseline: {}", self.bench);
        let _write_result = writeln!(summary);
        let _write_result = writeln!(summary, "| Field | Value |");
        let _write_result = writeln!(summary, "| --- | --- |");
        let _write_result = writeln!(summary, "| Outcome | {} |", self.outcome.as_str());
        let _write_result = writeln!(summary, "| Stage | {} |", self.stage);
        let _write_result = writeln!(
            summary,
            "| Reference revision | {} |",
            self.reference_revision.as_deref().unwrap_or("none")
        );
        let _write_result = writeln!(
            summary,
            "| Current revision | {} |",
            self.current_revision.as_deref().unwrap_or("unknown")
        );
        let _write_result = writeln!(summary, "| Threshold | {:.4} |", self.threshold);
        let _write_result = writeln!(summary, "| Comparable ids | {} |", self.comparable_count);
        if let Some(reason) = &self.reason {
            let _write_result = writeln!(summary, "| Reason | {reason} |");
        }
        if let Some(stage) = &self.failed_stage {
            let _write_result = writeln!(summary, "| Failed stage | {stage} |");
        }

        if !self.cases.is_empty() {
            let _write_result = writeln!(summary);
            let _write_result = writeln!(summary, "| Benchmark id | Class | Change interval |");
            let _write_result = writeln!(summary, "| --- | --- | --- |");
            for case in &self.cases {
                let _write_result = writeln!(
                    summary,
                    "| {} | {} | {} |",
                    case.id,
                    case.class.as_str(),
                    render_interval(case),
                );
            }
        }

        summary
    }

    /// Render the verdict as a single-line summary for a log or annotation.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{BaselineVerdict, Outcome};
    ///
    /// let verdict = BaselineVerdict::without_comparison(
    ///     "hnsw",
    ///     Outcome::NoReference,
    ///     "classify",
    ///     "shallow checkout",
    ///     0.05,
    /// );
    /// assert!(verdict.to_line().contains("outcome=no_reference"));
    /// ```
    #[must_use]
    pub fn to_line(&self) -> String {
        format!(
            "outcome={} stage={} comparable={} regressed={} threshold={:.4}",
            self.outcome.as_str(),
            self.stage,
            self.comparable_count,
            self.regressed_ids().len(),
            self.threshold,
        )
    }
}

/// Render one case's change interval, or the reason it has none.
fn render_interval(case: &CaseVerdict) -> String {
    match (case.lower_bound, case.upper_bound) {
        (Some(lower), Some(upper)) => format!("{lower:+.4} .. {upper:+.4}"),
        _ => case
            .reason
            .clone()
            .unwrap_or_else(|| "not compared".to_owned()),
    }
}

/// Count cases by class for diagnostics.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::{BaselineVerdict, CaseClass};
///
/// let counts = chutoro_test_support::ci::benchmark_baseline::verdict::class_counts(
///     &BaselineVerdict::started("hnsw", 0.05),
/// );
/// assert!(counts.is_empty());
/// ```
#[must_use]
pub fn class_counts(verdict: &BaselineVerdict) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for case in &verdict.cases {
        *counts.entry(case.class.as_str()).or_insert(0) += 1;
    }

    counts
}
