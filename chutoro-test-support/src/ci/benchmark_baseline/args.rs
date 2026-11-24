//! Parse the benchmark baseline helper's command line.
//!
//! The helper has one subcommand per lane stage: `inventory` records what a
//! measurement wrote, `stage` transfers reference data into the current tree,
//! and `classify` applies the regression policy to the comparison's change
//! estimates. Parsing is a pure function over the argument vector so every
//! spelling can be tested without spawning a process or touching the
//! filesystem.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::BaselineError;

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

impl ClassifyArgs {
    /// Build arguments for a normal classification.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::args::ClassifyArgs;
    ///
    /// let args = ClassifyArgs::new(
    ///     "/evidence/preflight.json".into(),
    ///     "hnsw",
    ///     "/evidence/verdict.json".into(),
    ///     0.05,
    /// );
    /// assert!(args.no_reference.is_none());
    /// ```
    #[must_use]
    pub fn new(
        preflight: std::path::PathBuf,
        bench: &str,
        verdict: std::path::PathBuf,
        threshold: f64,
    ) -> Self {
        Self {
            preflight,
            bench: bench.to_owned(),
            verdict,
            threshold,
            no_reference: None,
            failed_stage: None,
            failure_reason: None,
        }
    }
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

/// Collected `--name value` options with repeatable values.
#[derive(Debug, Default)]
struct Options {
    /// Option values, keyed by their long name.
    values: BTreeMap<String, Vec<String>>,
}

impl Options {
    /// Parse `--name value` pairs, rejecting unknown or misplaced arguments.
    fn parse(arguments: &[String]) -> Result<Self, BaselineError> {
        let mut options = Self::default();
        let mut remaining = arguments.iter();
        while let Some(argument) = remaining.next() {
            let name = if Self::is_known(argument) {
                argument.clone()
            } else if argument.starts_with('-') {
                return Err(usage(&format!("unknown option `{argument}`")));
            } else {
                return Err(usage(&format!(
                    "unexpected positional argument `{argument}`"
                )));
            };
            let value = remaining
                .next()
                .ok_or_else(|| usage(&format!("option `{name}` requires a value")))?;
            options.values.entry(name).or_default().push(value.clone());
        }

        Ok(options)
    }

    /// Return whether `name` is an option this helper understands.
    fn is_known(name: &str) -> bool {
        matches!(
            name,
            "--workspace"
                | "--bench"
                | "--baseline"
                | "--revision"
                | "--output"
                | "--fresh-since"
                | "--bench-argument"
                | "--reference-manifest"
                | "--current-manifest"
                | "--evidence"
                | "--filter-output"
                | "--preflight"
                | "--verdict"
                | "--threshold"
                | "--no-reference"
                | "--failed-stage"
                | "--failure-reason"
                | "--help"
        )
    }

    /// Return the first value supplied for `name`.
    fn get(&self, name: &str) -> Option<&String> {
        self.values.get(name).and_then(|values| values.first())
    }

    /// Return a required option value.
    fn required(&self, name: &str) -> Result<String, BaselineError> {
        self.get(name)
            .cloned()
            .ok_or_else(|| usage(&format!("option `{name}` is required")))
    }

    /// Return every recorded measurement argument in the order supplied.
    fn bench_arguments(&self) -> Vec<String> {
        self.values
            .get("--bench-argument")
            .cloned()
            .unwrap_or_default()
    }

    /// Return a required option value as a path.
    fn required_path(&self, name: &str) -> Result<PathBuf, BaselineError> {
        self.required(name).map(PathBuf::from)
    }

    /// Return an optional option value as a path.
    fn path(&self, name: &str) -> Option<PathBuf> {
        self.get(name).map(PathBuf::from)
    }
}

/// Build a usage failure.
fn usage(message: &str) -> BaselineError {
    BaselineError::Usage {
        message: message.to_owned(),
        remedy: USAGE_REMEDY,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for helper argument parsing.

    use super::{ClassifyArgs, Command, InventoryArgs, NoReferenceReason, StageArgs, parse};
    use crate::ci::benchmark_baseline::BaselineError;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn no_subcommand_is_a_usage_error() {
        let error = parse(&[]).expect_err("an empty command line must fail");

        assert!(matches!(error, BaselineError::Usage { .. }));
    }

    #[test]
    fn help_is_recognized() {
        assert_eq!(
            parse(&arguments(&["--help"])).expect("help parses"),
            Command::Help
        );
    }

    #[test]
    fn unknown_subcommand_names_the_subcommand() {
        let error = parse(&arguments(&["summarize"])).expect_err("unknown subcommand must fail");

        assert!(error.to_string().contains("summarize"), "{error}");
    }

    #[test]
    fn inventory_collects_required_options_and_repeatable_arguments() {
        let parsed = parse(&arguments(&[
            "inventory",
            "--workspace",
            "/work",
            "--bench",
            "hnsw",
            "--baseline",
            "ci-reference",
            "--revision",
            "abc123",
            "--output",
            "/evidence/reference.json",
            "--bench-argument",
            "--noplot",
            "--bench-argument",
            "--color",
        ]))
        .expect("inventory parses");

        let Command::Inventory(args) = parsed else {
            panic!("expected inventory");
        };
        let expected: InventoryArgs = InventoryArgs {
            workspace: "/work".into(),
            bench: "hnsw".to_owned(),
            baseline: "ci-reference".to_owned(),
            revision: "abc123".to_owned(),
            output: "/evidence/reference.json".into(),
            fresh_since: None,
            bench_arguments: vec!["--noplot".to_owned(), "--color".to_owned()],
        };
        assert_eq!(*args, expected);
    }

    #[test]
    fn inventory_rejects_a_missing_value() {
        let error =
            parse(&arguments(&["inventory", "--workspace"])).expect_err("missing value must fail");

        assert!(error.to_string().contains("--workspace"), "{error}");
    }

    #[test]
    fn unknown_option_is_a_usage_error() {
        let error = parse(&arguments(&["inventory", "--nope", "value"]))
            .expect_err("unknown option must fail");

        assert!(error.to_string().contains("--nope"), "{error}");
    }

    #[test]
    fn positional_argument_is_a_usage_error() {
        let error =
            parse(&arguments(&["inventory", "stray"])).expect_err("positional argument must fail");

        assert!(error.to_string().contains("stray"), "{error}");
    }

    #[test]
    fn stage_collects_manifests_and_evidence_directory() {
        let parsed = parse(&arguments(&[
            "stage",
            "--bench",
            "hnsw",
            "--reference-manifest",
            "/evidence/reference.json",
            "--current-manifest",
            "/evidence/current.json",
            "--evidence",
            "/evidence",
            "--filter-output",
            "/evidence/filter.txt",
        ]))
        .expect("stage parses");

        let Command::Stage(args) = parsed else {
            panic!("expected stage");
        };
        let expected: StageArgs = StageArgs {
            bench: "hnsw".to_owned(),
            reference_manifest: "/evidence/reference.json".into(),
            current_manifest: "/evidence/current.json".into(),
            evidence: "/evidence".into(),
            filter_output: Some("/evidence/filter.txt".into()),
        };
        assert_eq!(*args, expected);
    }

    #[test]
    fn classify_defaults_the_threshold() {
        let parsed = parse(&arguments(&[
            "classify",
            "--preflight",
            "/evidence/preflight.json",
            "--bench",
            "hnsw",
            "--verdict",
            "/evidence/verdict.json",
        ]))
        .expect("classify parses");

        let Command::Classify(args) = parsed else {
            panic!("expected classify");
        };
        assert_eq!(args.threshold, super::super::DEFAULT_REGRESSION_THRESHOLD);
        assert!(args.no_reference.is_none());
    }

    #[test]
    fn classify_accepts_an_explicit_threshold_and_reason() {
        let parsed = parse(&arguments(&[
            "classify",
            "--preflight",
            "/evidence/preflight.json",
            "--bench",
            "hnsw",
            "--verdict",
            "/evidence/verdict.json",
            "--threshold",
            "0.25",
            "--no-reference",
            "shallow",
        ]))
        .expect("classify parses");

        let Command::Classify(args) = parsed else {
            panic!("expected classify");
        };
        let expected: ClassifyArgs = ClassifyArgs {
            preflight: "/evidence/preflight.json".into(),
            bench: "hnsw".to_owned(),
            verdict: "/evidence/verdict.json".into(),
            threshold: 0.25,
            no_reference: Some(NoReferenceReason::Shallow),
            failed_stage: None,
            failure_reason: None,
        };
        assert_eq!(*args, expected);
    }

    #[test]
    fn classify_rejects_a_negative_threshold() {
        let error = parse(&arguments(&[
            "classify",
            "--preflight",
            "/evidence/preflight.json",
            "--bench",
            "hnsw",
            "--verdict",
            "/evidence/verdict.json",
            "--threshold",
            "-0.1",
        ]))
        .expect_err("negative threshold must fail");

        assert!(error.to_string().contains("--threshold"), "{error}");
    }

    #[test]
    fn classify_rejects_an_unknown_no_reference_reason() {
        let error = parse(&arguments(&[
            "classify",
            "--preflight",
            "/evidence/preflight.json",
            "--bench",
            "hnsw",
            "--verdict",
            "/evidence/verdict.json",
            "--no-reference",
            "detached",
        ]))
        .expect_err("unknown reason must fail");

        assert!(error.to_string().contains("detached"), "{error}");
    }

    #[test]
    fn unused_options_are_collected_but_ignored() {
        // `--unused` is not an option name, so this proves the parser rejects
        // any name outside the known set rather than silently accepting it.
        let error = parse(&arguments(&["classify", "--unused", "value"]))
            .expect_err("unknown option must fail");
        assert!(matches!(error, BaselineError::Usage { .. }));
    }
}
