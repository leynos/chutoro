//! End-to-end tests for the benchmark baseline helper against real Criterion.
//!
//! These tests build an isolated fixture crate with the workspace's pinned
//! Criterion 0.5.1, save a reference and a current baseline into separate
//! Criterion roots, stage the reference data into the current tree, run the
//! real `--load-baseline … --baseline …` comparison, and classify the result.
//! They are the acceptance evidence that the lane can execute a comparison end
//! to end rather than failing before it starts.
//!
//! The shared fixture driver lives in `benchmark_baseline_end_to_end/support.rs`.

#[path = "benchmark_baseline_end_to_end/support.rs"]
mod support;

use std::fs;

use chutoro_test_support::ci::benchmark_baseline::args::StageArgs;
use chutoro_test_support::ci::benchmark_baseline::compatibility;
use chutoro_test_support::ci::benchmark_baseline::preflight::PreflightReport;
use chutoro_test_support::ci::benchmark_baseline::runner::{PREFLIGHT_FILE, run_stage};
use chutoro_test_support::ci::benchmark_baseline::transfer::{StageRequest, stage};
use chutoro_test_support::ci::benchmark_baseline::{CaseClass, Outcome};
use support::{
    classify, compare_baseline, inventory_tree, measure_pair, save_baseline, stage_reference,
    temporary_root, write_manifests,
};

/// Threshold the fixture tests apply.
///
/// The fixture is a synthetic busy loop measured on a shared, often heavily
/// loaded host. A 5% bound is not stable there, and the acceptance these tests
/// establish is that the comparison path runs end to end and distinguishes a
/// benign result from a large, deliberate slowdown. The production default
/// remains 0.05 and is exercised by the unit and CLI tests.
const FIXTURE_THRESHOLD: f64 = 0.25;

#[test]
fn unchanged_fixture_passes_a_real_comparison() {
    let root = temporary_root().expect("temporary root must be created");
    let evidence = root.path().join("evidence");
    fs::create_dir_all(&evidence).expect("evidence directory must be created");
    let (reference, current) =
        measure_pair(root.path(), 1).expect("the fixture must measure both revisions");
    assert_eq!(reference.cases.len(), 3);
    assert!(
        reference
            .cases
            .iter()
            .all(|case| case.required_files.is_complete())
    );

    let (reference_path, current_path) =
        write_manifests(&evidence, &reference, &current).expect("manifests must be written");
    let filter = stage_reference(&evidence, &reference_path, &current_path)
        .expect("a filter must be emitted");
    assert!(filter.starts_with("^("));
    assert!(filter.ends_with(")$"));

    let current_home = root.path().join("current-criterion");
    compare_baseline(&current_home, &filter, 1).expect("the comparison must run");

    for case in &current.cases {
        let change = current_home
            .join(&case.relative_directory)
            .join("change/estimates.json");
        assert!(
            change.is_file(),
            "comparison must write {}",
            change.display()
        );
    }

    let verdict = classify(&evidence, FIXTURE_THRESHOLD).expect("classification must succeed");
    assert_eq!(verdict.outcome, Outcome::Pass);
    assert_eq!(verdict.comparable_count, 3);
    assert!(
        verdict
            .cases
            .iter()
            .all(|case| { matches!(case.class, CaseClass::WithinNoise | CaseClass::Improved) })
    );
    let summary = verdict.to_markdown();
    assert!(summary.contains("| Outcome | pass |"), "{summary}");
}

#[test]
fn controlled_slowdown_is_detected_as_a_regression() {
    let root = temporary_root().expect("temporary root must be created");
    let evidence = root.path().join("evidence");
    fs::create_dir_all(&evidence).expect("evidence directory must be created");
    let (reference, current) =
        measure_pair(root.path(), 3).expect("the fixture must measure both revisions");

    let (reference_path, current_path) =
        write_manifests(&evidence, &reference, &current).expect("manifests must be written");
    let filter = stage_reference(&evidence, &reference_path, &current_path)
        .expect("a filter must be emitted");
    let current_home = root.path().join("current-criterion");
    compare_baseline(&current_home, &filter, 3).expect("the comparison must run");

    let verdict = classify(&evidence, FIXTURE_THRESHOLD).expect("classification must succeed");
    assert_eq!(verdict.outcome, Outcome::Regression);
    assert_eq!(verdict.regressed_ids().len(), 3);
    assert!(
        verdict
            .cases
            .iter()
            .all(|case| case.class == CaseClass::Regressed)
    );
}

#[test]
fn current_only_identifier_is_unverified_and_the_strict_filter_does_not_panic() {
    let root = temporary_root().expect("temporary root must be created");
    let evidence = root.path().join("evidence");
    fs::create_dir_all(&evidence).expect("evidence directory must be created");
    let (mut reference, current) =
        measure_pair(root.path(), 1).expect("the fixture must measure both revisions");
    let removed = reference
        .cases
        .pop()
        .expect("the fixture must register at least one case");

    let (reference_path, current_path) =
        write_manifests(&evidence, &reference, &current).expect("manifests must be written");
    let filter = stage_reference(&evidence, &reference_path, &current_path)
        .expect("a filter must be emitted");
    assert!(!filter.contains(&removed.full_id), "{filter}");
    let current_home = root.path().join("current-criterion");
    // `--baseline ci-reference` is strict: if the filter selected the
    // current-only identifier, Criterion would panic on its missing baseline.
    compare_baseline(&current_home, &filter, 1).expect("the comparison must run");

    let verdict = classify(&evidence, FIXTURE_THRESHOLD).expect("classification must succeed");
    assert_eq!(verdict.outcome, Outcome::Pass);
    assert_eq!(verdict.comparable_count, 2);
    let unverified: Vec<&str> = verdict
        .cases
        .iter()
        .filter(|case| case.class == CaseClass::NewUnverified)
        .map(|case| case.id.as_str())
        .collect();
    assert_eq!(unverified, vec![removed.full_id.as_str()]);
}

#[test]
fn staging_rejects_a_reference_tree_with_a_deleted_baseline() {
    let root = temporary_root().expect("temporary root must be created");
    let evidence = root.path().join("evidence");
    fs::create_dir_all(&evidence).expect("evidence directory must be created");
    let reference_home = root.path().join("reference-criterion");
    let current_home = root.path().join("current-criterion");
    save_baseline(&reference_home, "ci-reference", 1).expect("reference must be saved");
    save_baseline(&current_home, "ci-current", 1).expect("current must be saved");
    let reference = inventory_tree(&reference_home, "ci-reference", "reference-revision")
        .expect("reference must be inventoried");
    let current = inventory_tree(&current_home, "ci-current", "current-revision")
        .expect("current must be inventoried");
    let victim = reference
        .cases
        .first()
        .expect("the fixture must register at least one case");
    fs::remove_dir_all(
        reference
            .output_root
            .join(&victim.relative_directory)
            .join("ci-reference"),
    )
    .expect("the reference baseline directory must be removed");
    let (reference_path, current_path) =
        write_manifests(&evidence, &reference, &current).expect("manifests must be written");

    let error = run_stage(&StageArgs {
        bench: "baseline".to_owned(),
        reference_manifest: reference_path,
        current_manifest: current_path,
        evidence: evidence.clone(),
        filter_output: None,
    })
    .expect_err("a deleted reference baseline must fail staging");

    let message = error.to_string();
    assert!(message.contains("ci-reference"), "{message}");
    let report = PreflightReport::read(&evidence.join(PREFLIGHT_FILE))
        .expect("a preflight report must be written on failure");
    assert_eq!(report.outcome, Outcome::Incomplete);
    let reason = report.reason.unwrap_or_default();
    assert!(reason.contains(&victim.full_id), "{reason}");
}

#[test]
fn stage_transfer_is_exercised_directly_for_nested_identifiers() {
    let root = temporary_root().expect("temporary root must be created");
    let reference_home = root.path().join("reference-criterion");
    let current_home = root.path().join("current-criterion");
    save_baseline(&reference_home, "ci-reference", 1).expect("reference must be saved");
    save_baseline(&current_home, "ci-current", 1).expect("current must be saved");
    let reference = inventory_tree(&reference_home, "ci-reference", "reference-revision")
        .expect("reference must be inventoried");
    let current = inventory_tree(&current_home, "ci-current", "current-revision")
        .expect("current must be inventoried");
    let compatibility =
        compatibility::check(&reference, &current).expect("the two trees must be compatible");

    let outcome = stage(&StageRequest {
        reference_root: &reference.output_root,
        current_root: &current.output_root,
        reference_baseline: "ci-reference",
        comparable: &compatibility.comparable,
    })
    .expect("the transfer must succeed");

    assert_eq!(outcome.staged.len(), 3);
    assert!(
        compatibility
            .comparable
            .iter()
            .any(|case| case.full_id == "fixture/alpha/with/slash")
    );
    for staged in &outcome.staged {
        assert!(staged.destination.join("sample.json").is_file());
    }
}
