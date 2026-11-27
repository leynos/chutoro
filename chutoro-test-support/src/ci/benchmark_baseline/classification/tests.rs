//! Unit tests for change-interval classification.

use std::fs;

use rstest::rstest;
use tempfile::TempDir;

use super::{
    CaseChange, ChangeEstimates, ConfidenceInterval, Estimate, change_estimates_path,
    classify_interval, read_case_change,
};
use crate::ci::benchmark_baseline::{BaselineError, CaseClass, REQUIRED_CONFIDENCE_LEVEL};

fn change(lower_bound: f64, upper_bound: f64) -> CaseChange {
    CaseChange {
        estimates: ChangeEstimates {
            mean: Estimate {
                confidence_interval: ConfidenceInterval {
                    confidence_level: REQUIRED_CONFIDENCE_LEVEL,
                    lower_bound,
                    upper_bound,
                },
                point_estimate: lower_bound,
            },
        },
    }
}

/// One finite interval and the class it must produce.
#[derive(Clone, Copy, Debug)]
struct IntervalCase {
    /// Lower bound of the mean change interval.
    lower: f64,
    /// Upper bound of the mean change interval.
    upper: f64,
    /// Class the policy must assign.
    class: CaseClass,
}

#[rstest]
#[case::lower_at_threshold(IntervalCase { lower: 0.05, upper: 0.30, class: CaseClass::WithinNoise })]
#[case::lower_above_threshold(IntervalCase { lower: 0.050_000_1, upper: 0.30, class: CaseClass::Regressed })]
#[case::upper_at_negative_threshold(IntervalCase { lower: -0.40, upper: -0.05, class: CaseClass::WithinNoise })]
#[case::upper_below_negative_threshold(IntervalCase { lower: -0.40, upper: -0.050_000_1, class: CaseClass::Improved })]
#[case::crossing_zero(IntervalCase { lower: -0.20, upper: 0.20, class: CaseClass::WithinNoise })]
#[case::equal_bounds(IntervalCase { lower: 0.20, upper: 0.20, class: CaseClass::Regressed })]
fn finite_intervals_are_classified_at_their_boundaries(#[case] case: IntervalCase) {
    let classification =
        classify_interval(case.lower, case.upper, 0.05).expect("ordered finite interval");

    assert_eq!(classification.class, case.class);
}

#[test]
fn a_non_finite_bound_is_rejected() {
    let error = classify_interval(f64::NAN, 0.30, 0.05).expect_err("NaN must not classify");

    assert!(matches!(error, super::IntervalError::NonFinite { .. }));
}

#[test]
fn a_non_finite_threshold_is_rejected() {
    let error =
        classify_interval(0.10, 0.30, f64::INFINITY).expect_err("infinite threshold must fail");

    assert!(matches!(error, super::IntervalError::NonFinite { .. }));
}

#[test]
fn a_non_positive_threshold_is_rejected() {
    let error = classify_interval(0.10, 0.30, 0.0).expect_err("a zero threshold must fail");

    assert!(matches!(error, super::IntervalError::Threshold { .. }));
}

#[test]
fn inverted_interval_is_rejected() {
    let error = classify_interval(0.30, 0.10, 0.05).expect_err("inverted interval must fail");

    assert_eq!(
        error,
        super::IntervalError::Inverted {
            lower_bound: 0.30,
            upper_bound: 0.10,
        }
    );
}

#[test]
fn case_classification_uses_the_recorded_interval() {
    let classification = change(0.30, 0.50)
        .classify("fixture/alpha", 0.05)
        .expect("valid change");

    assert_eq!(classification.class, CaseClass::Regressed);
    assert_eq!(classification.lower_bound, 0.30);
    assert_eq!(classification.upper_bound, 0.50);
}

#[test]
fn different_confidence_level_is_incompatible() {
    let mut case = change(0.30, 0.50);
    case.estimates.mean.confidence_interval.confidence_level = 0.99;

    let error = case
        .classify("fixture/alpha", 0.05)
        .expect_err("0.99 interval must fail");

    assert!(matches!(error, BaselineError::ConfidenceLevel { .. }));
    let message = error.to_string();
    assert!(message.contains("0.99"), "{message}");
    assert!(message.contains("0.95"), "{message}");
}

#[test]
fn inverted_recorded_interval_is_incomplete() {
    let error = change(0.50, 0.30)
        .classify("fixture/alpha", 0.05)
        .expect_err("inverted interval must fail");

    assert!(matches!(error, BaselineError::InvertedInterval { .. }));
    assert!(error.to_string().contains("fixture/alpha"), "{error}");
}

#[test]
fn missing_change_estimates_name_the_path() {
    let directory = TempDir::new().expect("temporary directory");
    let root = directory.path();

    let error = read_case_change(root, "fixture/alpha", "fixture/alpha")
        .expect_err("missing estimates must fail");

    assert!(matches!(error, BaselineError::MissingChange { .. }));
    let message = error.to_string();
    assert!(message.contains("change/estimates.json"), "{message}");
}

#[test]
fn malformed_change_estimates_are_incomplete() {
    let directory = TempDir::new().expect("temporary directory");
    let root = directory.path();
    let path = change_estimates_path(root, "fixture/alpha");
    fs::create_dir_all(path.parent().expect("path has a parent"))
        .expect("change directory must be created");
    fs::write(&path, "{not json").expect("malformed estimates must be written");

    let error = read_case_change(root, "fixture/alpha", "fixture/alpha")
        .expect_err("malformed estimates must fail");

    assert!(matches!(error, BaselineError::MalformedChange { .. }));
    assert!(error.to_string().contains("fixture/alpha"), "{error}");
}

#[test]
fn well_formed_change_estimates_are_read() {
    let directory = TempDir::new().expect("temporary directory");
    let root = directory.path();
    let path = change_estimates_path(root, "fixture/alpha");
    fs::create_dir_all(path.parent().expect("path has a parent"))
        .expect("change directory must be created");
    fs::write(
        &path,
        r#"{"mean":{"confidence_interval":{"confidence_level":0.95,"lower_bound":0.2,"upper_bound":0.4},"point_estimate":0.3,"standard_error":0.01}}"#,
    )
    .expect("estimates must be written");

    let change = read_case_change(root, "fixture/alpha", "fixture/alpha")
        .expect("valid estimates must be read");
    let classification = change
        .classify("fixture/alpha", 0.05)
        .expect("valid interval must classify");

    assert_eq!(classification.class, CaseClass::Regressed);
    assert_eq!(change.estimates.mean.point_estimate, 0.3);
}
