//! Classify Criterion's relative change estimates against a regression policy.
//!
//! Criterion writes `change/estimates.json` for each compared identifier. Its
//! `mean` entry holds the relative change and a confidence interval. Criterion
//! itself reports a change as significant using a t-test and a noise
//! threshold; that judgement is not a policy, so this module applies an
//! explicit rule instead: only a mean-change interval that lies wholly above
//! the threshold is a regression. An interval that straddles the threshold is
//! within noise, because the evidence does not exclude either side.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::verdict::CaseClass;
use super::{BaselineError, REQUIRED_CONFIDENCE_LEVEL};

/// The confidence interval Criterion recorded for one estimate.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct Estimate {
    /// Confidence level the interval was computed at.
    pub confidence_interval: ConfidenceInterval,
    /// Point estimate of the relative change.
    pub point_estimate: f64,
}

/// One confidence interval from a Criterion estimate.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct ConfidenceInterval {
    /// Confidence level the interval was computed at.
    pub confidence_level: f64,
    /// Lower bound of the interval.
    pub lower_bound: f64,
    /// Upper bound of the interval.
    pub upper_bound: f64,
}

/// The subset of Criterion's `change/estimates.json` this helper reads.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct ChangeEstimates {
    /// Relative change in the mean.
    pub mean: Estimate,
}

/// Classification of one interval under the configured threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Classification {
    /// Verdict class assigned to the identifier.
    pub class: CaseClass,
    /// Lower bound of the mean change interval.
    pub lower_bound: f64,
    /// Upper bound of the mean change interval.
    pub upper_bound: f64,
}

/// Classify a mean change interval under a two-sided threshold.
///
/// A lower bound strictly above `threshold` is a regression; an upper bound
/// strictly below `-threshold` is an improvement; anything else is within
/// noise.
///
/// # Errors
///
/// Returns [`IntervalError`] when the bounds are inverted.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::classification::classify_interval;
/// use chutoro_test_support::ci::benchmark_baseline::CaseClass;
///
/// let regressed = classify_interval(0.12, 0.30, 0.05).expect("ordered interval");
/// assert_eq!(regressed.class, CaseClass::Regressed);
///
/// let noise = classify_interval(-0.01, 0.20, 0.05).expect("ordered interval");
/// assert_eq!(noise.class, CaseClass::WithinNoise);
/// ```
#[expect(
    clippy::float_arithmetic,
    reason = "the policy is defined over Criterion's floating-point change interval"
)]
pub fn classify_interval(
    lower_bound: f64,
    upper_bound: f64,
    threshold: f64,
) -> Result<Classification, IntervalError> {
    if lower_bound > upper_bound {
        return Err(IntervalError {
            lower_bound,
            upper_bound,
        });
    }
    let negative_threshold = -threshold;
    let class = if lower_bound > threshold {
        CaseClass::Regressed
    } else if upper_bound < negative_threshold {
        CaseClass::Improved
    } else {
        CaseClass::WithinNoise
    };

    Ok(Classification {
        class,
        lower_bound,
        upper_bound,
    })
}

/// An interval whose bounds are inverted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalError {
    /// Lower bound as supplied.
    pub lower_bound: f64,
    /// Upper bound as supplied.
    pub upper_bound: f64,
}

/// The relative change and interval recorded for one identifier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaseChange {
    /// Change estimates read from `change/estimates.json`.
    pub estimates: ChangeEstimates,
}

impl CaseChange {
    /// Classify this identifier's mean change interval.
    ///
    /// # Errors
    ///
    /// Returns an error when the interval used a different confidence level or
    /// when its bounds are inverted.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::classification::{
    ///     CaseChange, ChangeEstimates, Classification, ConfidenceInterval, Estimate,
    /// };
    /// use chutoro_test_support::ci::benchmark_baseline::CaseClass;
    ///
    /// let change = CaseChange {
    ///     estimates: ChangeEstimates {
    ///         mean: Estimate {
    ///             confidence_interval: ConfidenceInterval {
    ///                 confidence_level: 0.95,
    ///                 lower_bound: 0.30,
    ///                 upper_bound: 0.40,
    ///             },
    ///             point_estimate: 0.35,
    ///         },
    ///     },
    /// };
    /// let classification = change.classify("fixture/alpha", 0.05).expect("valid interval");
    /// assert_eq!(classification.class, CaseClass::Regressed);
    /// ```
    pub fn classify(&self, id: &str, threshold: f64) -> Result<Classification, BaselineError> {
        let interval = self.estimates.mean.confidence_interval;
        require_confidence_level(id, interval.confidence_level)?;
        classify_interval(interval.lower_bound, interval.upper_bound, threshold).map_err(|error| {
            BaselineError::InvertedInterval {
                id: id.to_owned(),
                lower: error.lower_bound,
                upper: error.upper_bound,
                remedy: "delete the malformed change estimates and rerun the comparison",
            }
        })
    }
}

/// Reject an estimate computed at a confidence level the policy did not assume.
#[expect(
    clippy::float_arithmetic,
    reason = "the confidence level is a float written by Criterion"
)]
fn require_confidence_level(id: &str, found: f64) -> Result<(), BaselineError> {
    let expected = REQUIRED_CONFIDENCE_LEVEL;
    let difference = (found - expected).abs();
    if difference < f64::EPSILON {
        return Ok(());
    }

    Err(BaselineError::ConfidenceLevel {
        id: id.to_owned(),
        found,
        expected,
        remedy: "compare with `--confidence-level 0.95`, the level this policy assumes",
    })
}

/// Read one identifier's change estimates from a Criterion output root.
///
/// # Errors
///
/// Returns an error when the comparison did not write change estimates, or
/// when they cannot be parsed.
///
/// # Examples
///
/// ```no_run
/// # use std::path::Path;
/// use chutoro_test_support::ci::benchmark_baseline::classification::read_case_change;
///
/// let change = read_case_change(
///     Path::new("target/criterion"),
///     "fixture/alpha",
///     "fixture/alpha",
/// )
/// .expect("the comparison wrote change estimates");
/// assert!(change.estimates.mean.point_estimate.is_finite());
/// ```
pub fn read_case_change(
    output_root: &Path,
    directory_name: &str,
    id: &str,
) -> Result<CaseChange, BaselineError> {
    let path = change_estimates_path(output_root, directory_name);
    let bytes = std::fs::read(&path).map_err(|source| BaselineError::MissingChange {
        id: id.to_owned(),
        path: path.clone(),
        remedy: "compare with `--load-baseline` and `--baseline` so Criterion writes change estimates",
    })?;
    let estimates: ChangeEstimates =
        serde_json::from_slice(&bytes).map_err(|source| BaselineError::MalformedChange {
            id: id.to_owned(),
            path: path.clone(),
            source,
            remedy: "delete the malformed change estimates and rerun the comparison",
        })?;

    Ok(CaseChange { estimates })
}

/// Return the path Criterion writes change estimates to.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use chutoro_test_support::ci::benchmark_baseline::classification::change_estimates_path;
///
/// assert_eq!(
///     change_estimates_path(Path::new("/criterion"), "fixture/alpha/n=1"),
///     Path::new("/criterion/fixture/alpha/n=1/change/estimates.json"),
/// );
/// ```
#[must_use]
pub fn change_estimates_path(output_root: &Path, directory_name: &str) -> PathBuf {
    output_root
        .join(directory_name)
        .join("change")
        .join("estimates.json")
}

#[cfg(test)]
mod tests;
