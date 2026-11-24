//! Parse the benchmark baseline helper's command line.
//!
//! The helper has one subcommand per lane stage: `inventory` records what a
//! measurement wrote, `stage` transfers reference data into the current tree,
//! and `classify` applies the regression policy to the comparison's change
//! estimates. Parsing is a pure function over the argument vector so every
//! spelling can be tested without spawning a process or touching the
//! filesystem.

use std::path::PathBuf;

use super::BaselineError;

mod options;
use options::Options;

/// Usage remedy shared by every argument failure.
const USAGE_REMEDY: &str = "run `benchmark_baseline --help` for the accepted arguments";

/// One parsed invocation of the helper.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Record a baseline tree's layout.
    Inventory(Box<InventoryArgs>),
    /// Transfer reference data into the current tree.
    Stage(Box<StageArgs>),
    /// Classify the comparison's change estimates.
    Classify(Box<ClassifyArgs>),
    /// Print usage and succeed.
    Help,
}

/// Arguments for `inventory`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InventoryArgs {
    /// Workspace whose tool versions are recorded.
    pub workspace: PathBuf,
    /// Benchmark target name.
    pub bench: String,
    /// Criterion baseline name.
    pub baseline: String,
    /// Revision the baseline was measured from.
    pub revision: String,
    /// Path the manifest is written to.
    pub output: PathBuf,
    /// Optional freshness marker the baseline must be newer than.
    pub fresh_since: Option<PathBuf>,
    /// Measurement arguments, excluding baseline selection.
    pub bench_arguments: Vec<String>,
}

/// Arguments for `stage`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageArgs {
    /// Benchmark target name.
    pub bench: String,
    /// Reference inventory manifest.
    pub reference_manifest: PathBuf,
    /// Current inventory manifest.
    pub current_manifest: PathBuf,
    /// Directory the preflight report is written to.
    pub evidence: PathBuf,
    /// Optional path the emitted filter is written to.
    pub filter_output: Option<PathBuf>,
}

/// Arguments for `classify`.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassifyArgs {
    /// Preflight report written by `stage`.
    pub preflight: PathBuf,
    /// Benchmark target name, for the verdict.
    pub bench: String,
    /// Verdict path inside the evidence directory.
    pub verdict: PathBuf,
    /// Regression threshold applied to the lower bound.
    pub threshold: f64,
    /// Why no reference existed, when none did.
    pub no_reference: Option<NoReferenceReason>,
    /// Stage that failed, when the workflow is recording a lane defect.
    pub failed_stage: Option<String>,
    /// Reason a stage failed, when the workflow is recording a lane defect.
    pub failure_reason: Option<String>,
}

/// Why the lane had no previous revision to compare against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoReferenceReason {
    /// The checkout did not include the previous commit.
    Shallow,
    /// The current commit is the repository's first.
    RootCommit,
}

impl NoReferenceReason {
    /// Return the stable lowercase name recorded in the verdict.
    ///
    ///
    /// # Returns
    ///
    /// The stable lowercase reason name.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::args::NoReferenceReason;
    ///
    /// assert_eq!(NoReferenceReason::Shallow.as_str(), "shallow");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shallow => "shallow",
            Self::RootCommit => "root-commit",
        }
    }

    /// Parse the recorded reason.
    ///
    ///
    /// # Parameters
    ///
    /// - `value`: the recorded reason string.
    ///
    /// # Returns
    ///
    /// The matching reason, or `None` for an unknown value.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::args::NoReferenceReason;
    ///
    /// assert_eq!(
    ///     NoReferenceReason::parse("root-commit"),
    ///     Some(NoReferenceReason::RootCommit),
    /// );
    /// ```
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "shallow" => Some(Self::Shallow),
            "root-commit" => Some(Self::RootCommit),
            _ => None,
        }
    }
}

/// Parse the helper's argument vector, excluding the program name.
///
/// # Errors
///
/// Returns [`BaselineError::Usage`] naming the missing, unknown, or malformed
/// argument.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::args::{Command, parse};
///
/// assert_eq!(parse(&["--help".to_owned()]).expect("help parses"), Command::Help);
/// assert!(parse(&[]).is_err());
/// ```
pub fn parse(arguments: &[String]) -> Result<Command, BaselineError> {
    let Some(subcommand) = arguments.first() else {
        return Err(usage("no subcommand was given"));
    };
    let rest = arguments.get(1..).unwrap_or_default();

    match subcommand.as_str() {
        "--help" | "-h" | "help" => Ok(Command::Help),
        "inventory" => parse_inventory(rest),
        "stage" => parse_stage(rest),
        "classify" => parse_classify(rest),
        other => Err(usage(&format!("unknown subcommand `{other}`"))),
    }
}

/// Parse `inventory` arguments.
fn parse_inventory(arguments: &[String]) -> Result<Command, BaselineError> {
    let options = Options::parse(arguments)?;
    let bench_arguments = options.bench_arguments();

    Ok(Command::Inventory(Box::new(InventoryArgs {
        workspace: options.required_path("--workspace")?,
        bench: options.required("--bench")?,
        baseline: options.required("--baseline")?,
        revision: options.required("--revision")?,
        output: options.required_path("--output")?,
        fresh_since: options.path("--fresh-since"),
        bench_arguments,
    })))
}

/// Parse `stage` arguments.
fn parse_stage(arguments: &[String]) -> Result<Command, BaselineError> {
    let options = Options::parse(arguments)?;

    Ok(Command::Stage(Box::new(StageArgs {
        bench: options.required("--bench")?,
        reference_manifest: options.required_path("--reference-manifest")?,
        current_manifest: options.required_path("--current-manifest")?,
        evidence: options.required_path("--evidence")?,
        filter_output: options.path("--filter-output"),
    })))
}

/// Parse `classify` arguments.
fn parse_classify(arguments: &[String]) -> Result<Command, BaselineError> {
    let options = Options::parse(arguments)?;
    let threshold = match options.get("--threshold") {
        Some(value) => value
            .parse::<f64>()
            .map_err(|error| usage(&format!("`--threshold {value}` is not a number: {error}")))?,
        None => super::DEFAULT_REGRESSION_THRESHOLD,
    };
    if !(threshold.is_finite() && threshold > 0.0) {
        return Err(usage(&format!(
            "`--threshold {threshold}` must be a finite positive fraction"
        )));
    }

    let no_reference = match options.get("--no-reference") {
        Some(value) => Some(NoReferenceReason::parse(value).ok_or_else(|| {
            usage(&format!(
                "`--no-reference {value}` must be `shallow` or `root-commit`"
            ))
        })?),
        None => None,
    };

    Ok(Command::Classify(Box::new(ClassifyArgs {
        preflight: options.required_path("--preflight")?,
        bench: options.required("--bench")?,
        verdict: options.required_path("--verdict")?,
        threshold,
        no_reference,
        failed_stage: options.get("--failed-stage").cloned(),
        failure_reason: options.get("--failure-reason").cloned(),
    })))
}

/// Build a usage failure.
pub(super) fn usage(message: &str) -> BaselineError {
    BaselineError::Usage {
        message: message.to_owned(),
        remedy: USAGE_REMEDY,
    }
}

#[cfg(test)]
mod tests;
