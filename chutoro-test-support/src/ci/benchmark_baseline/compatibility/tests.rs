//! Unit tests for benchmark baseline compatibility rules.

use std::collections::BTreeMap;
use std::path::Path;

use super::check;
use crate::ci::benchmark_baseline::{
    BaselineCase, BaselineError, BaselineManifest, Outcome, RequiredFiles, ToolVersions,
};

fn case(full_id: &str, complete: bool) -> BaselineCase {
    case_with_files(full_id, complete, complete)
}

fn case_with_files(full_id: &str, estimates: bool, sample: bool) -> BaselineCase {
    BaselineCase {
        full_id: full_id.to_owned(),
        directory_name: full_id.to_owned(),
        relative_directory: full_id.into(),
        required_files: RequiredFiles {
            benchmark: true,
            estimates,
            sample,
        },
    }
}

fn manifest(baseline: &str, cases: Vec<BaselineCase>) -> BaselineManifest {
    BaselineManifest {
        schema_version: 1,
        revision: "abc".to_owned(),
        bench: "hnsw".to_owned(),
        baseline: baseline.to_owned(),
        output_root: Path::new("/work/target/criterion").to_path_buf(),
        tool_versions: ToolVersions {
            criterion: "0.5.1".to_owned(),
            cargo: "cargo 1.99.1".to_owned(),
            rustc: "rustc 1.99.1".to_owned(),
        },
        bench_arguments: vec!["--noplot".to_owned()],
        environment: BTreeMap::new(),
        cases,
    }
}

fn reference(cases: Vec<BaselineCase>) -> BaselineManifest {
    manifest("ci-reference", cases)
}

fn current(cases: Vec<BaselineCase>) -> BaselineManifest {
    manifest("ci-current", cases)
}

#[test]
fn shared_identifiers_are_comparable() {
    let compatibility = check(
        &reference(vec![case("fixture/alpha", true)]),
        &current(vec![case("fixture/alpha", true)]),
    )
    .expect("shared identifiers are compatible");

    assert_eq!(compatibility.comparable.len(), 1);
    assert_eq!(compatibility.new_unverified, Vec::<String>::new());
    assert_eq!(compatibility.removed, Vec::<String>::new());
}

#[test]
fn current_only_identifiers_are_new_unverified() {
    let compatibility = check(
        &reference(vec![case("fixture/alpha", true)]),
        &current(vec![
            case("fixture/alpha", true),
            case("fixture/beta", true),
        ]),
    )
    .expect("a current-only identifier is reported, not fatal");

    assert_eq!(compatibility.new_unverified, vec!["fixture/beta"]);
}

#[test]
fn reference_only_identifiers_are_removed() {
    let compatibility = check(
        &reference(vec![
            case("fixture/alpha", true),
            case("fixture/beta", true),
        ]),
        &current(vec![case("fixture/alpha", true)]),
    )
    .expect("a reference-only identifier is reported, not fatal");

    assert_eq!(compatibility.removed, vec!["fixture/beta"]);
}

#[test]
fn incompatible_criterion_versions_are_named() {
    let mut reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let mut current_manifest = current(vec![case("fixture/alpha", true)]);
    reference_manifest.tool_versions.criterion = "0.5.0".to_owned();
    current_manifest.tool_versions.criterion = "0.5.1".to_owned();

    let error = check(&reference_manifest, &current_manifest)
        .expect_err("different Criterion versions cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    let message = error.to_string();
    assert!(message.contains("criterion version support"), "{message}");
}

#[test]
fn unsupported_criterion_version_is_incompatible() {
    let mut reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let mut current_manifest = current(vec![case("fixture/alpha", true)]);
    reference_manifest.tool_versions.criterion = "0.6.0".to_owned();
    current_manifest.tool_versions.criterion = "0.6.0".to_owned();

    let error = check(&reference_manifest, &current_manifest)
        .expect_err("an unsupported Criterion version cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    assert!(error.to_string().contains("0.6.0"), "{error}");
}

#[test]
fn different_rust_toolchains_are_incompatible() {
    let mut reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let mut current_manifest = current(vec![case("fixture/alpha", true)]);
    reference_manifest.tool_versions.rustc = "rustc 1.99.1".to_owned();
    current_manifest.tool_versions.rustc = "rustc 1.100.0".to_owned();

    let error = check(&reference_manifest, &current_manifest)
        .expect_err("different toolchains cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    assert!(error.to_string().contains("rustc -Vv"), "{error}");
}

#[test]
fn different_benchmark_targets_are_incompatible() {
    let reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let mut current_manifest = current(vec![case("fixture/alpha", true)]);
    current_manifest.bench = "mst".to_owned();

    let error = check(&reference_manifest, &current_manifest)
        .expect_err("different targets cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    assert!(error.to_string().contains("benchmark target"), "{error}");
}

#[test]
fn different_measurement_arguments_are_incompatible() {
    let reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let mut current_manifest = current(vec![case("fixture/alpha", true)]);
    current_manifest.bench_arguments = vec!["--quick".to_owned()];

    let error = check(&reference_manifest, &current_manifest)
        .expect_err("different arguments cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    assert!(error.to_string().contains("benchmark arguments"), "{error}");
}

#[test]
fn differing_environment_values_name_the_key() {
    let reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let mut current_manifest = current(vec![case("fixture/alpha", true)]);
    current_manifest.environment.insert(
        "CHUTORO_BENCH_HNSW_MEMORY_PROFILE".to_owned(),
        "1".to_owned(),
    );

    let error = check(&reference_manifest, &current_manifest)
        .expect_err("different environment values cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    assert!(
        error
            .to_string()
            .contains("CHUTORO_BENCH_HNSW_MEMORY_PROFILE"),
        "{error}"
    );
}

#[test]
fn identical_baseline_names_are_incompatible() {
    let reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let current_manifest = reference(vec![case("fixture/alpha", true)]);

    let error = check(&reference_manifest, &current_manifest)
        .expect_err("staging into the current baseline must be refused");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    assert!(error.to_string().contains("baseline names"), "{error}");
}

#[test]
fn empty_comparable_set_is_incomplete() {
    let error = check(
        &reference(vec![case("fixture/alpha", true)]),
        &current(vec![case("fixture/beta", true)]),
    )
    .expect_err("no shared identifier cannot produce a comparison");

    assert_eq!(error.outcome(), Some(Outcome::Incomplete));
    assert!(matches!(error, BaselineError::EmptyComparable { .. }));
    // The empty-comparable message names both baseline names, because the
    // identifiers differ and there is no single one to point at.
    assert!(error.to_string().contains("ci-reference"), "{error}");
    assert!(error.to_string().contains("ci-current"), "{error}");
}

#[test]
fn missing_reference_sample_is_incomplete_and_names_the_path() {
    let error = check(
        &reference(vec![case_with_files("fixture/alpha", true, false)]),
        &current(vec![case("fixture/alpha", true)]),
    )
    .expect_err("missing reference sample cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incomplete));
    let message = error.to_string();
    assert!(message.contains("sample.json"), "{message}");
    assert!(message.contains("ci-reference"), "{message}");
}

#[test]
fn missing_reference_estimates_are_incomplete_and_name_the_path() {
    let error = check(
        &reference(vec![case_with_files("fixture/alpha", false, true)]),
        &current(vec![case("fixture/alpha", true)]),
    )
    .expect_err("missing reference estimates cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incomplete));
    let message = error.to_string();
    assert!(message.contains("estimates.json"), "{message}");
}

#[test]
fn duplicate_identifiers_are_incomplete_and_name_both_paths() {
    let error = check(
        &reference(vec![
            case("fixture/alpha", true),
            case("fixture/alpha", true),
        ]),
        &current(vec![case("fixture/alpha", true)]),
    )
    .expect_err("duplicate identifiers cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incomplete));
    assert!(matches!(error, BaselineError::DuplicateId { .. }));
    let message = error.to_string();
    assert!(message.contains("benchmark.json"), "{message}");
}

#[test]
fn unsupported_schema_is_incompatible() {
    let mut reference_manifest = reference(vec![case("fixture/alpha", true)]);
    reference_manifest.schema_version = 99;
    let current_manifest = current(vec![case("fixture/alpha", true)]);

    let error = check(&reference_manifest, &current_manifest)
        .expect_err("an unknown schema cannot be compared");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    assert!(error.to_string().contains("schema"), "{error}");
}
