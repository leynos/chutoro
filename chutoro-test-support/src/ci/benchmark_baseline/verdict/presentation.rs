//! Markdown and single-line rendering for a [`BaselineVerdict`](super::BaselineVerdict).

use std::collections::BTreeMap;

use super::{BaselineVerdict, CaseVerdict};

impl BaselineVerdict {
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
    #[expect(
        clippy::format_push_string,
        reason = "the summary is assembled from many small labelled rows"
    )]
    pub fn to_markdown(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("## Benchmark baseline: {}\n\n", self.bench));
        summary.push_str("| Field | Value |\n| --- | --- |\n");
        summary.push_str(&format!("| Outcome | {} |\n", self.outcome.as_str()));
        summary.push_str(&format!("| Stage | {} |\n", escape_cell(&self.stage)));
        summary.push_str(&format!(
            "| Reference revision | {} |\n",
            escape_cell(self.reference_revision.as_deref().unwrap_or("none"))
        ));
        summary.push_str(&format!(
            "| Current revision | {} |\n",
            escape_cell(self.current_revision.as_deref().unwrap_or("unknown"))
        ));
        summary.push_str(&format!("| Threshold | {:.4} |\n", self.threshold));
        summary.push_str(&format!("| Comparable ids | {} |\n", self.comparable_count));
        if let Some(reason) = &self.reason {
            summary.push_str(&format!("| Reason | {} |\n", escape_cell(reason)));
        }
        if let Some(stage) = &self.failed_stage {
            summary.push_str(&format!("| Failed stage | {} |\n", escape_cell(stage)));
        }

        if !self.cases.is_empty() {
            summary.push_str("\n| Benchmark id | Class | Change interval |\n| --- | --- | --- |\n");
            for case in &self.cases {
                summary.push_str(&format!(
                    "| {} | {} | {} |\n",
                    escape_cell(&case.id),
                    case.class.as_str(),
                    escape_cell(&render_interval(case)),
                ));
            }
        }

        summary
    }

    /// Render the verdict as a single-line summary for a log or annotation.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::verdict::{
    ///     BaselineVerdict, NoComparison,
    /// };
    /// use chutoro_test_support::ci::benchmark_baseline::Outcome;
    ///
    /// let verdict = BaselineVerdict::without_comparison(&NoComparison {
    ///     bench: "hnsw",
    ///     outcome: Outcome::NoReference,
    ///     stage: "classify",
    ///     reason: "shallow checkout",
    ///     threshold: 0.05,
    /// });
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

/// Make a value safe for one Markdown table cell.
///
/// A pipe would open a new column and a line break would end the row, so both
/// are escaped. Only rendering is affected; the JSON evidence keeps the
/// original value.
fn escape_cell(value: &str) -> String {
    value
        .replace('|', "\\|")
        .replace("\r\n", " ")
        .replace(['\n', '\r'], " ")
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
