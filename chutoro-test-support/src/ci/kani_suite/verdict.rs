//! Compare a parsed Kani run with the harnesses the manifest expects.
//!
//! Kani exits successfully after verifying fewer harnesses than were asked
//! for, and a selector that matches nothing is indistinguishable in the exit
//! status from a selector that matched its proof. The comparison here is the
//! part that makes a reduced run fail: the exact multiset of successfully
//! completed harnesses must equal the expected set, with no failure, no
//! duplicate and no unexpected completion, and the run's own summary must
//! agree with the blocks it printed.

use std::collections::BTreeMap;

use thiserror::Error;

use super::output::{Outcome, SuiteReport, Summary};

/// A reason a run's result does not discharge the expected proof surface.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerdictError {
    /// No harness completed successfully.
    #[error("Kani completed no harness successfully")]
    Empty,
    /// Harnesses reported a failed or undetermined outcome.
    #[error("Kani reported a failed or undetermined outcome for: {names:?}")]
    Failed {
        /// The harnesses that did not verify.
        names: Vec<String>,
    },
    /// The same harness was reported more than once.
    #[error("Kani reported these harnesses more than once: {names:?}")]
    Duplicate {
        /// The repeated harnesses.
        names: Vec<String>,
    },
    /// An expected harness did not complete successfully.
    #[error("Kani did not complete these expected harnesses successfully: {names:?}")]
    Missing {
        /// The expected harnesses absent from the successful set.
        names: Vec<String>,
    },
    /// A harness outside the requested scope completed.
    #[error("Kani completed harnesses outside the requested scope: {names:?}")]
    Unexpected {
        /// The harnesses that should not have run.
        names: Vec<String>,
    },
    /// The summary's totals disagree with the parsed blocks.
    #[error(
        "the manual harness summary reports {summary_succeeded} successes, \
         {summary_failed} failures and {summary_total} total, but the parsed \
         blocks hold {parsed_succeeded} successes, {parsed_failed} failures and \
         {parsed_total} total"
    )]
    SummaryMismatch {
        /// Successes the summary claims.
        summary_succeeded: usize,
        /// Failures the summary claims.
        summary_failed: usize,
        /// Total harnesses the summary claims.
        summary_total: usize,
        /// Successful blocks actually parsed.
        parsed_succeeded: usize,
        /// Failed blocks actually parsed.
        parsed_failed: usize,
        /// Blocks actually parsed.
        parsed_total: usize,
    },
    /// The summary named a failure no block reported.
    #[error("the manual harness summary names these failures with no failed block: {names:?}")]
    UnreportedFailure {
        /// Failure names the blocks did not record.
        names: Vec<String>,
    },
    /// The output carried no `Complete - ...` line at all.
    #[error(
        "Kani's output carries no manual harness summary; a complete run ends \
         with its totals, so a truncated or redirected record cannot be accepted"
    )]
    MissingSummary,
}

/// Require `report` to hold exactly the expected successful harness multiset.
///
/// `expected` is a set: the manifest lists each identity once, and a delegated
/// command may not select one twice. A successful harness reported twice is a
/// duplicate and fails, because a second result for one proof means the run is
/// not the one the expected set describes.
///
/// # Errors
///
/// Returns [`VerdictError`] when a harness failed, a result is duplicated, an
/// expected harness is missing, an unexpected harness completed, or the
/// summary disagrees with the parsed blocks.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeSet;
///
/// use chutoro_test_support::ci::kani_suite::output::parse;
/// use chutoro_test_support::ci::kani_suite::verdict::evaluate;
///
/// let report = parse(
///     "Checking harness proofs::verify_one...\n\
///      VERIFICATION:- SUCCESSFUL\n\
///      Manual Harness Summary:\n\
///      Complete - 1 successfully verified harnesses, 0 failures, 1 total.\n",
/// )
/// .expect("a complete run");
/// let expected = BTreeSet::from(["proofs::verify_one".to_owned()]);
/// assert!(evaluate(&expected, &report).is_ok());
/// ```
pub fn evaluate(
    expected: &std::collections::BTreeSet<String>,
    report: &SuiteReport,
) -> Result<(), VerdictError> {
    let successful = report.successful();
    let duplicates = repeated(&successful);
    if !duplicates.is_empty() {
        return Err(VerdictError::Duplicate { names: duplicates });
    }
    let failed = report.unsuccessful();
    if !failed.is_empty() {
        return Err(VerdictError::Failed {
            names: failed.into_iter().map(str::to_owned).collect(),
        });
    }
    if successful.is_empty() {
        return Err(VerdictError::Empty);
    }
    let missing = expected
        .iter()
        .filter(|name| !successful.contains(&name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(VerdictError::Missing { names: missing });
    }
    let unexpected = successful
        .iter()
        .filter(|name| !expected.contains(**name))
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(VerdictError::Unexpected { names: unexpected });
    }
    check_summary(report, &successful)
}

/// Require the manual harness summary, then cross-check it against the blocks.
///
/// The summary is what distinguishes a complete run from a truncated one: the
/// per-harness blocks arrive as they finish, so a record cut off after the
/// last expected block would otherwise look complete. A run with no
/// `Complete - ...` line is therefore rejected rather than accepted on its
/// blocks alone.
fn check_summary(report: &SuiteReport, successful: &[&str]) -> Result<(), VerdictError> {
    let Some(summary) = report.summary else {
        return Err(VerdictError::MissingSummary);
    };
    let parsed_total = report.results.len();
    let parsed_succeeded = successful.len();
    let parsed_failed = parsed_total - parsed_succeeded;
    let parsed = Summary {
        succeeded: parsed_succeeded,
        failed: parsed_failed,
        total: parsed_total,
    };
    if summary != parsed {
        return Err(VerdictError::SummaryMismatch {
            summary_succeeded: summary.succeeded,
            summary_failed: summary.failed,
            summary_total: summary.total,
            parsed_succeeded,
            parsed_failed,
            parsed_total,
        });
    }
    let failed = report
        .results
        .iter()
        .filter(|result| result.outcome == Outcome::Failed)
        .map(|result| result.name.as_str())
        .collect::<Vec<_>>();
    let unrecorded = report
        .reported_failures
        .iter()
        .filter(|name| !failed.contains(&name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unrecorded.is_empty() {
        return Err(VerdictError::UnreportedFailure { names: unrecorded });
    }
    Ok(())
}

/// Return the names appearing more than once in `names`, sorted.
fn repeated(names: &[&str]) -> Vec<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for name in names {
        *counts.entry(name).or_default() += 1;
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(name, _)| name.to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    //! Unit tests for the expected-result comparison.

    use std::collections::BTreeSet;

    use super::*;
    use crate::ci::kani_suite::output::HarnessResult;

    /// Build a report holding `names` with the supplied outcome.
    ///
    /// The summary matches the blocks, because a run without one is now a
    /// rejection: every case that is about something else has to be a complete
    /// record, or it would fail on the summary before reaching its own check.
    fn report_with(names: &[&str], outcome: Outcome) -> SuiteReport {
        let mut report = SuiteReport {
            results: names
                .iter()
                .map(|name| HarnessResult {
                    name: (*name).to_owned(),
                    outcome,
                })
                .collect(),
            summary: None,
            reported_failures: Vec::new(),
        };
        let failed = names.iter().filter(|_| outcome != Outcome::Success).count();
        report.summary = Some(Summary {
            succeeded: names.len() - failed,
            failed,
            total: names.len(),
        });
        report
    }

    /// Build the expected set from string slices.
    fn expected(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn accepts_the_exact_expected_set() {
        let report = report_with(&["a::verify_one", "a::verify_two"], Outcome::Success);
        assert!(evaluate(&expected(&["a::verify_one", "a::verify_two"]), &report).is_ok());
    }

    #[test]
    fn rejects_a_missing_harness() {
        let report = report_with(&["a::verify_one"], Outcome::Success);
        assert_eq!(
            evaluate(&expected(&["a::verify_one", "a::verify_two"]), &report),
            Err(VerdictError::Missing {
                names: vec!["a::verify_two".to_owned()],
            })
        );
    }

    #[test]
    fn rejects_an_unexpected_harness() {
        let report = report_with(&["a::verify_one", "a::verify_two"], Outcome::Success);
        assert_eq!(
            evaluate(&expected(&["a::verify_one"]), &report),
            Err(VerdictError::Unexpected {
                names: vec!["a::verify_two".to_owned()],
            })
        );
    }

    #[test]
    fn rejects_a_failed_harness() {
        let report = report_with(&["a::verify_one"], Outcome::Failed);
        assert_eq!(
            evaluate(&expected(&["a::verify_one"]), &report),
            Err(VerdictError::Failed {
                names: vec!["a::verify_one".to_owned()],
            })
        );
    }

    #[test]
    fn rejects_a_duplicate_result() {
        let report = report_with(&["a::verify_one", "a::verify_one"], Outcome::Success);
        assert_eq!(
            evaluate(&expected(&["a::verify_one"]), &report),
            Err(VerdictError::Duplicate {
                names: vec!["a::verify_one".to_owned()],
            })
        );
    }

    #[test]
    fn rejects_an_empty_run() {
        let report = SuiteReport::default();
        assert_eq!(
            evaluate(&expected(&["a::verify_one"]), &report),
            Err(VerdictError::Empty)
        );
    }

    #[test]
    fn rejects_a_summary_that_disagrees_with_the_blocks() {
        let mut report = report_with(&["a::verify_one"], Outcome::Success);
        report.summary = Some(Summary {
            succeeded: 2,
            failed: 0,
            total: 2,
        });
        assert!(matches!(
            evaluate(&expected(&["a::verify_one"]), &report),
            Err(VerdictError::SummaryMismatch { .. })
        ));
    }

    #[test]
    fn rejects_a_record_with_no_summary() {
        let mut report = report_with(&["a::verify_one"], Outcome::Success);
        report.summary = None;
        assert_eq!(
            evaluate(&expected(&["a::verify_one"]), &report),
            Err(VerdictError::MissingSummary)
        );
    }

    #[test]
    fn rejects_a_summary_failure_with_no_failed_block() {
        let mut report = report_with(&["a::verify_one"], Outcome::Success);
        report.summary = Some(Summary {
            succeeded: 1,
            failed: 0,
            total: 1,
        });
        report.reported_failures.push("a::verify_two".to_owned());
        assert_eq!(
            evaluate(&expected(&["a::verify_one"]), &report),
            Err(VerdictError::UnreportedFailure {
                names: vec!["a::verify_two".to_owned()],
            })
        );
    }
}
