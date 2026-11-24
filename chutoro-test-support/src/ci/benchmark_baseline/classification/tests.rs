//! Unit tests for change-interval classification.

use std::fs;

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

#[test]
fn lower_bound_at_the_threshold_is_within_noise() {
    let classification = classify_interval(0.05, 0.30, 0.05).expect("ordered interval");

    assert_eq!(classification.class, CaseClass::WithinNoise);
}

#[test]
fn lower_bound_strictly_above_the_threshold_is_regressed() {
    let classification = classify_interval(0.050_000_1, 0.30, 0.05).expect("ordered interval");

    assert_eq!(classification.class, CaseClass::Regressed);
}

#[test]
fn upper_bound_at_the_negative_threshold_is_within_noise() {
    let classification = classify_interval(-0.40, -0.05, 0.05).expect("ordered interval");

    assert_eq!(classification.class, CaseClass::WithinNoise);
}

#[test]
fn upper_bound_strictly_below_the_negative_threshold_is_improved() {
    let classification = classify_interval(-0.40, -0.050_000_1, 0.05).expect("ordered interval");

    assert_eq!(classification.class, CaseClass::Improved);
}

#[test]
fn interval_crossing_zero_is_within_noise() {
    let classification = classify_interval(-0.20, 0.20, 0.05).expect("ordered interval");

    assert_eq!(classification.class, CaseClass::WithinNoise);
}

#[test]
fn interval_with_equal_bounds_is_accepted() {
    let classification = classify_interval(0.20, 0.20, 0.05).expect("ordered interval");

    assert_eq!(classification.class, CaseClass::Regressed);
}

#[test]
fn inverted_interval_is_rejected() {
    let error = classify_interval(0.30, 0.10, 0.05).expect_err("inverted interval must fail");

    assert_eq!(error.lower_bound, 0.30);
    assert_eq!(error.upper_bound, 0.10);
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
