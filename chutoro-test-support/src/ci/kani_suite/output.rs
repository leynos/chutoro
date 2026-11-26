//! Parse the regular-format output of one Kani run.
//!
//! Kani prints one `Checking harness <name>...` line before verifying a
//! harness and one `VERIFICATION:- <status>` line when it finishes. The
//! regular output format ends with a `Manual Harness Summary:` block and a
//! `Complete - N successfully verified harnesses, M failures, T total.` line.
//! Those anchors are what make a run checkable: a harness whose block has no
//! outcome never finished, and a summary that disagrees with the blocks means
//! the log is not a complete record of the run.
//!
//! Parsing fails closed. An outcome line with no open block, a second
//! `Checking harness` while a block is open, or end of output with a block
//! still open are all errors, because each is evidence that the run was
//! interrupted or that its output was not the format this checker pins.
//!
//! [`super::line`] answers the per-line questions this state machine asks;
//! this module owns the run's progress and the report it produces.

use thiserror::Error;

use super::line;

/// The outcome Kani reported for one harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// `VERIFICATION:- SUCCESSFUL`.
    Success,
    /// `VERIFICATION:- FAILED`.
    Failed,
    /// Any other verdict, such as an undetermined run.
    Other,
}

/// One completed harness block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessResult {
    /// The fully qualified harness path Kani printed.
    pub name: String,
    /// The outcome Kani reported.
    pub outcome: Outcome,
}

/// The totals from a `Manual Harness Summary` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    /// Successfully verified harnesses.
    pub succeeded: usize,
    /// Failed harnesses.
    pub failed: usize,
    /// Harnesses in total.
    pub total: usize,
}

/// The parsed result of one Kani run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SuiteReport {
    /// Every completed harness block, in output order.
    pub results: Vec<HarnessResult>,
    /// The summary totals, when the output carried them.
    pub summary: Option<Summary>,
    /// Harnesses the summary named as failed.
    pub reported_failures: Vec<String>,
}

impl SuiteReport {
    /// Return the names of every harness that completed successfully.
    #[must_use]
    pub fn successful(&self) -> Vec<&str> {
        self.results
            .iter()
            .filter(|result| result.outcome == Outcome::Success)
            .map(|result| result.name.as_str())
            .collect()
    }

    /// Return the names of every harness that did not verify successfully.
    #[must_use]
    pub fn unsuccessful(&self) -> Vec<&str> {
        self.results
            .iter()
            .filter(|result| result.outcome != Outcome::Success)
            .map(|result| result.name.as_str())
            .collect()
    }
}

/// A reason the output is not a complete, parseable Kani record.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OutputError {
    /// A new block opened before the previous one reported an outcome.
    #[error(
        "Kani started harness {next:?} before completing {open:?}; the run was \
         interrupted or the output was interleaved"
    )]
    Incomplete {
        /// The harness whose block never closed.
        open: String,
        /// The harness that started next.
        next: String,
    },
    /// End of output arrived with a block still open.
    #[error("Kani never reported an outcome for harness {open:?}")]
    Unterminated {
        /// The harness whose block never closed.
        open: String,
    },
    /// An outcome line appeared with no block open.
    #[error("Kani reported the outcome {outcome:?} with no harness open")]
    OutcomeWithoutBlock {
        /// The outcome line's status text.
        outcome: String,
    },
    /// Kani found no proof harnesses at all.
    #[error("Kani found no proof harnesses to verify")]
    NoHarnesses,
    /// Kani printed automatically generated harnesses this checker does not
    /// compare against the manifest.
    #[error("Kani reported an autoharness, which this checker does not model")]
    Autoharness,
    /// The output was written per harness rather than to stdout.
    #[error("Kani output is not the regular format: {line:?}")]
    Unrecognised {
        /// The line that could not be classified.
        line: String,
    },
}

/// Parse one Kani run's regular-format stdout.
///
/// # Errors
///
/// Returns [`OutputError`] when a block is incomplete, an outcome appears
/// without a block, the output records no harness, or the output carries an
/// unsupported harness kind.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::kani_suite::output::{Outcome, parse};
///
/// let report = parse(
///     "Checking harness proofs::verify_one...\n\
///      SUMMARY:\n ** 0 of 3 failed\n\n\
///      VERIFICATION:- SUCCESSFUL\n\
///      Manual Harness Summary:\n\
///      Complete - 1 successfully verified harnesses, 0 failures, 1 total.\n",
/// )
/// .expect("a complete run");
/// assert_eq!(report.results[0].name, "proofs::verify_one");
/// assert_eq!(report.results[0].outcome, Outcome::Success);
/// assert_eq!(report.summary.expect("summary").total, 1);
/// ```
pub fn parse(output: &str) -> Result<SuiteReport, OutputError> {
    let mut run = Run::default();
    for raw in output.lines().map(line::strip_ansi) {
        run.line(raw.trim_end())?;
    }
    run.finish()
}

/// One run's parser state: the report so far and the block it is inside.
#[derive(Default)]
struct Run {
    /// The report accumulated from completed blocks.
    report: SuiteReport,
    /// The harness whose block is open, if any.
    open: Option<String>,
    /// Whether the manual summary block has started.
    in_summary: bool,
}

impl Run {
    /// Fold one output line into the run's state.
    fn line(&mut self, line: &str) -> Result<(), OutputError> {
        if let Some(name) = line::checking_harness(line) {
            return self.start(name);
        }
        if line.starts_with("Autoharness: Checking function ") {
            return Err(OutputError::Autoharness);
        }
        if let Some(status) = line.strip_prefix("VERIFICATION:- ") {
            let Some(name) = self.open.take() else {
                return Err(OutputError::OutcomeWithoutBlock {
                    outcome: status.to_owned(),
                });
            };
            self.complete(name, line::outcome_of(status));
            return Ok(());
        }
        self.marker(line)
    }

    /// Open a block for `name`, rejecting a block that never closed.
    fn start(&mut self, name: String) -> Result<(), OutputError> {
        if let Some(open) = self.open.take() {
            return Err(OutputError::Incomplete { open, next: name });
        }
        self.open = Some(name);
        Ok(())
    }

    /// Record the block closed for `name` with the outcome Kani reported.
    fn complete(&mut self, name: String, outcome: Outcome) {
        self.report.results.push(HarnessResult { name, outcome });
    }

    /// Fold a summary marker or a summary line into the report.
    fn marker(&mut self, line: &str) -> Result<(), OutputError> {
        if line == "Manual Harness Summary:" {
            self.in_summary = true;
            return Ok(());
        }
        if line == "No proof harnesses (functions with #[kani::proof]) were found to verify." {
            return Err(OutputError::NoHarnesses);
        }
        if self.in_summary {
            line::read_summary_line(line, &mut self.report);
        }
        Ok(())
    }

    /// Return the finished report, rejecting output that ended mid-block.
    fn finish(self) -> Result<SuiteReport, OutputError> {
        match self.open {
            Some(open) => Err(OutputError::Unterminated { open }),
            None => Ok(self.report),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the Kani output parser.

    use super::*;

    #[test]
    fn parses_two_complete_harnesses() {
        let output = concat!(
            "Checking harness a::verify_one...\n",
            "SUMMARY:\n ** 0 of 3 failed\n\n",
            "VERIFICATION:- SUCCESSFUL\n",
            "Checking harness a::verify_two...\n",
            "SUMMARY:\n ** 1 of 4 failed\n\n",
            "VERIFICATION:- FAILED\n",
            "Manual Harness Summary:\n",
            "Verification failed for - a::verify_two\n",
            "Complete - 1 successfully verified harnesses, 1 failures, 2 total.\n",
        );
        let report = parse(output).expect("two complete blocks");
        assert_eq!(report.successful(), vec!["a::verify_one"]);
        assert_eq!(report.unsuccessful(), vec!["a::verify_two"]);
        assert_eq!(
            report.summary,
            Some(Summary {
                succeeded: 1,
                failed: 1,
                total: 2,
            })
        );
        assert_eq!(report.reported_failures, vec!["a::verify_two"]);
    }

    #[test]
    fn rejects_a_block_that_never_finished() {
        let output = concat!(
            "Checking harness a::verify_one...\n",
            "Checking harness a::verify_two...\n",
            "VERIFICATION:- SUCCESSFUL\n",
        );
        assert_eq!(
            parse(output),
            Err(OutputError::Incomplete {
                open: "a::verify_one".to_owned(),
                next: "a::verify_two".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_output_that_ends_mid_block() {
        let output = "Checking harness a::verify_one...\n";
        assert_eq!(
            parse(output),
            Err(OutputError::Unterminated {
                open: "a::verify_one".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_an_outcome_with_no_block() {
        assert!(matches!(
            parse("VERIFICATION:- SUCCESSFUL\n"),
            Err(OutputError::OutcomeWithoutBlock { .. })
        ));
    }

    #[test]
    fn rejects_an_empty_proof_surface() {
        let output = concat!(
            "Manual Harness Summary:\n",
            "No proof harnesses (functions with #[kani::proof]) were found to verify.\n",
        );
        assert_eq!(parse(output), Err(OutputError::NoHarnesses));
    }

    #[test]
    fn strips_colour_from_a_verdict() {
        let output = concat!(
            "Checking harness a::verify_one...\n",
            "VERIFICATION:- \u{1b}[32mSUCCESSFUL\u{1b}[0m\n",
            "Manual Harness Summary:\n",
            "Complete - 1 successfully verified harnesses, 0 failures, 1 total.\n",
        );
        let report = parse(output).expect("a coloured verdict parses");
        assert_eq!(report.successful(), vec!["a::verify_one"]);
    }
}
