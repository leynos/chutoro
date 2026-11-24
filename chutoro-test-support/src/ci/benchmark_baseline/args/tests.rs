//! Unit tests for helper argument parsing.

use super::{ClassifyArgs, Command, InventoryArgs, NoReferenceReason, StageArgs, parse};
use crate::ci::benchmark_baseline::BaselineError;

fn arguments(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

/// The `classify` arguments shared by the option-rejection tests.
const CLASSIFY_BASE: [&str; 6] = [
    "classify",
    "--preflight",
    "/evidence/preflight.json",
    "--bench",
    "hnsw",
    "--verdict",
];

/// Build a `classify` argument list from the shared base and extra options.
fn classify_arguments(extra: &[&str]) -> Vec<String> {
    let mut values: Vec<String> = CLASSIFY_BASE
        .iter()
        .map(|value| (*value).to_owned())
        .collect();
    values.push("/evidence/verdict.json".to_owned());
    values.extend(extra.iter().map(|value| (*value).to_owned()));
    values
}

/// Parse a vector with the `inventory` subcommand and require a usage error.
fn parse_error(values: &[&str], expectation: &str) -> BaselineError {
    parse(&arguments(values)).expect_err(expectation)
}

/// Parse a full argument list and require a usage error.
fn parse_usage_error(values: &[&str], expectation: &str) -> BaselineError {
    parse(&arguments(values)).expect_err(expectation)
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
    let error = parse_error(&["summarize"], "unknown subcommand must fail");

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
    let error = parse_error(&["inventory", "--workspace"], "missing value must fail");

    assert!(error.to_string().contains("--workspace"), "{error}");
}

#[test]
fn unknown_option_is_a_usage_error() {
    let error = parse_error(
        &["inventory", "--nope", "value"],
        "unknown option must fail",
    );

    assert!(error.to_string().contains("--nope"), "{error}");
}

#[test]
fn positional_argument_is_a_usage_error() {
    let error = parse_error(&["inventory", "stray"], "positional argument must fail");

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
    let error = parse(&classify_arguments(&["--threshold", "-0.1"]))
        .expect_err("negative threshold must fail");

    assert!(error.to_string().contains("--threshold"), "{error}");
}

#[test]
fn classify_rejects_an_unknown_no_reference_reason() {
    let error = parse(&classify_arguments(&["--no-reference", "detached"]))
        .expect_err("unknown reason must fail");

    assert!(error.to_string().contains("detached"), "{error}");
}

#[test]
fn unused_options_are_collected_but_ignored() {
    // `--unused` is not an option name, so this proves the parser rejects
    // any name outside the known set rather than silently accepting it.
    let error = parse_usage_error(
        &["classify", "--unused", "value"],
        "unknown option must fail",
    );
    assert!(matches!(error, BaselineError::Usage { .. }));
}
