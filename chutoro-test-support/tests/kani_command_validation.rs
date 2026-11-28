//! Contract tests for the delegated-command validator's refusal surface.
//!
//! `kani_suite` refuses a delegated `cargo kani` command it cannot trust: one
//! that hides or redirects the output, selects a binary target, drops the
//! library or `--exact` requirement, repeats a selector, or is not a Cargo
//! Kani invocation at all. Each refusal is the only thing standing between a
//! run and a verdict it did not earn, so every branch is asserted here rather
//! than left to the CLI tests, which exercise the common paths only.
//!
//! The cases call the public [`parse_command`], which is the same entry point
//! the checker uses, so a removed check fails these tests even when the
//! command-line front end still compiles.
//!
//! Run via `make test`.

use chutoro_test_support::ci::kani_suite::runner::{CommandError, parse_command};
use rstest::rstest;

/// Return a case's words as owned arguments.
fn owned(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

/// Return the words of one valid invocation, ready to be extended.
fn valid_words() -> Vec<&'static str> {
    vec!["cargo", "kani", "-p", "pkg", "--lib", "--exact"]
}

#[rstest]
#[case("--quiet")]
#[case("-q")]
#[case("--output-into-files")]
#[case("--no-codegen")]
#[case("--only-codegen")]
#[case("--log-file")]
#[case("--tests")]
#[case("-j")]
#[case("--jobs")]
fn every_forbidden_flag_is_refused(#[case] flag: &str) {
    let mut words = valid_words();
    words.push(flag);
    match parse_command(&owned(&words), "pkg") {
        Err(CommandError::Forbidden { flag: named, .. }) => assert_eq!(named, flag),
        other => panic!("{flag} must be refused, got {other:?}"),
    }
}

#[rstest]
#[case("-j4", "-j")]
#[case("--jobs=4", "--jobs")]
#[case("--log-file=run.log", "--log-file")]
#[case("--output-into-files=dir", "--output-into-files")]
fn a_forbidden_option_is_refused_in_its_attached_spelling(
    #[case] argument: &str,
    #[case] expected: &str,
) {
    // `-j4` is the short option with its value attached; an exact-match
    // comparison would let it reach the child and interleave the blocks the
    // checker reads.
    let mut words = valid_words();
    words.push(argument);
    match parse_command(&owned(&words), "pkg") {
        Err(CommandError::Forbidden { flag, .. }) => assert_eq!(flag, expected),
        other => panic!("{argument} must be refused, got {other:?}"),
    }
}

#[rstest]
#[case(&["--output-format", "json"])]
#[case(&["--output-format=json"])]
fn a_non_regular_output_format_is_refused(#[case] extra: &[&str]) {
    let mut words = valid_words();
    words.extend(extra.iter().copied());
    match parse_command(&owned(&words), "pkg") {
        Err(CommandError::OutputFormat { format }) => assert_eq!(format, "json"),
        other => panic!("a non-regular format must be refused, got {other:?}"),
    }
}

#[test]
fn the_regular_output_format_is_accepted() {
    let mut words = valid_words();
    words.extend(["--output-format", "regular"]);
    assert!(
        parse_command(&owned(&words), "pkg").is_ok(),
        "the regular format is the one the parser pins"
    );
}

#[rstest]
#[case("--output-format")]
#[case("--harness")]
#[case("-p")]
#[case("--bin")]
fn an_option_without_its_value_is_refused(#[case] flag: &str) {
    let mut words = valid_words();
    words.push(flag);
    match parse_command(&owned(&words), "pkg") {
        Err(CommandError::MissingValue { flag: named }) => assert_eq!(named, flag),
        other => panic!("{flag} without a value must be refused, got {other:?}"),
    }
}

#[test]
fn a_command_that_is_not_cargo_kani_is_refused() {
    let outcome = parse_command(&owned(&["cargo", "build"]), "pkg");
    assert!(
        matches!(outcome, Err(CommandError::NotCargoKani { .. })),
        "a non-Kani command must be refused, got {outcome:?}"
    );
}

#[test]
fn an_empty_command_is_refused() {
    assert_eq!(parse_command(&[], "pkg"), Err(CommandError::Empty));
}

#[test]
fn a_command_that_drops_the_library_requirement_is_refused() {
    let words = ["cargo", "kani", "-p", "pkg", "--exact", "--harness", "a"];
    assert_eq!(
        parse_command(&owned(&words), "pkg"),
        Err(CommandError::MissingLib)
    );
}

#[test]
fn selectors_without_exact_are_refused() {
    // Without `--exact` Kani treats a selector as a substring, so a renamed
    // proof could still match and the expected set would describe another run.
    let words = ["cargo", "kani", "-p", "pkg", "--lib", "--harness", "a"];
    assert_eq!(
        parse_command(&owned(&words), "pkg"),
        Err(CommandError::NotExact)
    );
}

#[test]
fn a_duplicate_selector_is_refused() {
    let words = [
        "cargo",
        "kani",
        "-p",
        "pkg",
        "--lib",
        "--exact",
        "--harness",
        "a",
        "--harness",
        "a",
    ];
    match parse_command(&owned(&words), "pkg") {
        Err(CommandError::DuplicateSelector { names }) => assert_eq!(names, vec!["a"]),
        other => panic!("a repeated selector must be refused, got {other:?}"),
    }
}

#[test]
fn a_binary_target_is_refused_in_both_spellings() {
    for extra in [
        ["--bin", "helper"].as_slice(),
        ["--bin=helper"].as_slice(),
        ["--bins"].as_slice(),
    ] {
        let mut words = valid_words();
        words.extend(extra.iter().copied());
        let outcome = parse_command(&owned(&words), "pkg");
        assert!(
            matches!(
                outcome,
                Err(CommandError::BinaryTarget { .. } | CommandError::AllBinaries)
            ),
            "the binary target in {extra:?} must be refused, got {outcome:?}"
        );
    }
}

#[rstest]
#[case("-vpother")]
#[case("-vppkg")]
fn a_compact_option_inside_a_short_cluster_is_refused(#[case] argument: &str) {
    // Cargo reads `-vpother` as `-v -p other`, so the word can select a
    // package the reader does not record. Refusing the cluster is what stops
    // a second package reaching the child while the checker believes the
    // command selected only the expected one.
    let mut words = valid_words();
    words.push(argument);
    match parse_command(&owned(&words), "pkg") {
        Err(CommandError::ClusteredOption { argument: refused }) => {
            assert_eq!(refused, argument);
        }
        other => panic!("{argument} must be refused, got {other:?}"),
    }
}

#[test]
fn a_value_bearing_short_option_with_an_unrelated_name_is_accepted() {
    // `-Zpolonius` and `-Cpanic=abort` take their value attached; they are
    // Cargo's own spelling, not a cluster that could hide a package.
    for argument in ["-Zpolonius", "-Cpanic=abort", "-vv"] {
        let mut words = valid_words();
        words.push(argument);
        assert!(
            parse_command(&owned(&words), "pkg").is_ok(),
            "{argument} must not be read as a clustered package option"
        );
    }
}

#[test]
fn a_second_package_is_refused() {
    let words = ["cargo", "kani", "-p", "pkg", "other", "--lib", "--exact"];
    assert!(
        matches!(
            parse_command(&owned(&words), "pkg"),
            Err(CommandError::Package { .. })
        ),
        "a second package must be refused"
    );
}
