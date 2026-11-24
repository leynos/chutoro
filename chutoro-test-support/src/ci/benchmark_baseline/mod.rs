//! Discover, stage, and classify Criterion benchmark baselines for CI.
//!
//! Criterion writes each benchmark's data beneath
//! `<output-root>/<benchmark-id>/<baseline-name>/`, where `<benchmark-id>` may
//! itself contain path separators for a group, function, and parameter. A
//! single `ci-reference` directory at the output root therefore never holds a
//! baseline, which is why the scheduled comparison lane failed before it could
//! compare anything. This module reads Criterion's real layout, transfers
//! reference data into the current tree's collision-adjusted identifier
//! directories, and classifies the resulting confidence interval.
//!
//! The helper is deliberately local to this repository. It moves to a shared
//! actions repository only after a second consumer demonstrates the same
//! need. See `docs/developers-guide.md`, "Benchmark regression workflow", and
//! `docs/chutoro-design.md` §11.6.
//!
//! # Examples
//!
//! ```
//! use chutoro_test_support::ci::benchmark_baseline::{
//!     DEFAULT_REGRESSION_THRESHOLD, Outcome, exit_code,
//! };
//!
//! assert_eq!(exit_code(Outcome::Pass), 0);
//! assert!(DEFAULT_REGRESSION_THRESHOLD > 0.0);
//! ```

pub mod args;
pub mod classification;
pub mod compatibility;
pub mod error;
pub mod manifest;
pub mod output_root;
pub mod preflight;
pub mod runner;
pub mod transfer;
pub mod verdict;

use serde::{Deserialize, Serialize};

pub use error::{BaselineDiagnostic, BaselineError};
pub use manifest::{BaselineCase, BaselineManifest, RequiredFiles, ToolVersions};
pub use preflight::{ComparableCase, ManifestSummary, PreflightReport};
pub use verdict::{BaselineVerdict, CaseClass, CaseVerdict};

/// Schema version written by this helper and required when reading it back.
pub const SCHEMA_VERSION: u32 = 1;

/// The Criterion release whose output layout this helper understands.
///
/// Criterion 0.5.1 resolves the output root from `CRITERION_HOME`, then
/// `CARGO_TARGET_DIR`, then `cargo metadata`, then `./target/criterion`, and
/// stores one directory per benchmark identifier. A different release must be
/// validated against its own analysis source before it is accepted here.
pub const SUPPORTED_CRITERION_VERSION: &str = "0.5.1";

/// Confidence level Criterion must have used for a readable comparison.
///
/// The mean change interval in `change/estimates.json` carries the configured
/// level. The helper requires exactly this value so a lane cannot silently
/// compare intervals produced under a different configuration.
pub const REQUIRED_CONFIDENCE_LEVEL: f64 = 0.95;

/// Default regression threshold applied to the mean change lower bound.
///
/// A lower bound strictly above this value is a regression. The threshold is a
/// project decision for the scheduled lane, not a Criterion default, and is
/// recorded in every verdict so a reader can reproduce the judgement.
pub const DEFAULT_REGRESSION_THRESHOLD: f64 = 0.05;

/// Exit code used by the helper for a successful comparison.
pub const EXIT_PASS: i32 = 0;

/// Exit code used when a comparable benchmark regressed.
pub const EXIT_REGRESSION: i32 = 1;

/// Exit code used when the lane could not produce comparable evidence.
pub const EXIT_INCOMPLETE: i32 = 2;

/// Exit code used when the two sides are not comparable.
pub const EXIT_INCOMPATIBLE: i32 = 3;

/// Exit code used when no reference revision could be compared.
pub const EXIT_NO_REFERENCE: i32 = 4;

/// Exit code used for invalid helper arguments.
pub const EXIT_USAGE: i32 = 64;

/// Outcome of one benchmark baseline comparison.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Every comparable identifier stayed within noise.
    Pass,
    /// At least one comparable identifier regressed.
    Regression,
    /// The lane could not produce comparable evidence.
    Incomplete,
    /// The two sides used incompatible tools, targets, or settings.
    Incompatible,
    /// No previous revision existed to compare against.
    NoReference,
}

impl Outcome {
    /// Return the stable lowercase name written to reports and summaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::Outcome;
    ///
    /// assert_eq!(Outcome::Pass.as_str(), "pass");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Regression => "regression",
            Self::Incomplete => "incomplete",
            Self::Incompatible => "incompatible",
            Self::NoReference => "no_reference",
        }
    }
}

/// Map a comparison outcome to the helper's documented exit code.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::{
///     EXIT_REGRESSION, Outcome, exit_code,
/// };
///
/// assert_eq!(exit_code(Outcome::Regression), EXIT_REGRESSION);
/// ```
#[must_use]
pub const fn exit_code(outcome: Outcome) -> i32 {
    match outcome {
        Outcome::Pass => EXIT_PASS,
        Outcome::Regression => EXIT_REGRESSION,
        Outcome::Incomplete => EXIT_INCOMPLETE,
        Outcome::Incompatible => EXIT_INCOMPATIBLE,
        Outcome::NoReference => EXIT_NO_REFERENCE,
    }
}

/// Environment variable prefix recorded for comparison compatibility.
pub(crate) const BENCH_ENV_PREFIX: &str = "CHUTORO_BENCH_";
