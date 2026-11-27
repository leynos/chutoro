//! Validate and run the delegated Kani command.
//!
//! The checker does not build the Kani invocation itself; it is given one
//! after `--`, so the Makefile stays the only place that decides flags,
//! packages and unwind bounds. That freedom needs a boundary: a delegated
//! command that suppresses its output, redirects it per harness, or widens its
//! own exit status would leave the checker comparing a run it cannot see. The
//! flags that do that are rejected here before the process starts.
//!
//! A leading `NAME=VALUE` word is an environment assignment for the delegated
//! process only. That is how `KANI_ENV` reaches `cargo kani` without being set
//! on the checker itself, which needs no part of the verifier's toolchain. The
//! execution itself lives in [`super::process`], which this module's types
//! feed.

use thiserror::Error;

use super::cargo_options;
use super::parser;

/// A Kani option the checker cannot work with, with the reason it is refused.
pub(super) const FORBIDDEN_FLAGS: &[(&str, &str)] = &[
    (
        "--quiet",
        "it suppresses the per-harness blocks the checker parses",
    ),
    (
        "-q",
        "it suppresses the per-harness blocks the checker parses",
    ),
    (
        "--output-into-files",
        "it writes the per-harness blocks away from stdout",
    ),
    (
        "--no-codegen",
        "it verifies nothing while still starting the run",
    ),
    ("--only-codegen", "it compiles without verifying"),
    (
        "--log-file",
        "it writes the per-harness blocks away from stdout",
    ),
    (
        "--tests",
        "it changes which functions are treated as harnesses",
    ),
    (
        "-j",
        "it interleaves the per-harness blocks this checker reads",
    ),
    (
        "--jobs",
        "it interleaves the per-harness blocks this checker reads",
    ),
];

/// The delegated Kani invocation, after validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delegated {
    /// Environment assignments applied to the child alone.
    pub env: Vec<(String, String)>,
    /// The program to run.
    pub program: String,
    /// The program's arguments, beginning with the `kani` subcommand.
    pub args: Vec<String>,
    /// The package the command selects.
    pub package: String,
    /// The harness selectors the command passes, if any.
    pub selectors: Vec<String>,
}

/// A reason a delegated command is not one the checker can trust.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CommandError {
    /// No command followed the argument separator.
    #[error("no Kani command follows '--'")]
    Empty,
    /// The command is not a cargo-kani invocation.
    #[error("the delegated command must run `kani` as its first argument, got {args:?}")]
    NotCargoKani {
        /// The arguments that were supplied.
        args: Vec<String>,
    },
    /// An option was supplied without its value.
    #[error("option {flag} requires a value")]
    MissingValue {
        /// The option that is missing a value.
        flag: String,
    },
    /// An option would hide or redirect the run.
    #[error("the delegated command passes {flag}, but {reason}")]
    Forbidden {
        /// The refused option.
        flag: String,
        /// Why it is refused.
        reason: &'static str,
    },
    /// The output format is not the regular one the parser pins.
    #[error(
        "the delegated command requests output format {format:?}; only \"regular\" can be checked"
    )]
    OutputFormat {
        /// The requested format.
        format: String,
    },
    /// The command does not name exactly the package the checker was asked for.
    #[error("the delegated command must select exactly package {expected:?}; it selects {found:?}")]
    Package {
        /// The package the checker was asked to verify.
        expected: String,
        /// The packages the command selects.
        found: Vec<String>,
    },
    /// The command does not restrict itself to the library target.
    #[error("the delegated command must pass --lib, so only the library's harnesses are compiled")]
    MissingLib,
    /// The command selects a binary target, whose harnesses Kani reports
    /// under the same names as the library's.
    #[error(
        "the delegated command selects the binary target {name:?}, but only the library's \
         harnesses can be attributed to the manifest"
    )]
    BinaryTarget {
        /// The binary target that was selected.
        name: String,
    },
    /// The command selects every binary target.
    #[error(
        "the delegated command passes --bins, but only the library's harnesses can be \
         attributed to the manifest"
    )]
    AllBinaries,
    /// A harness selector is a substring filter rather than an exact name.
    #[error(
        "the delegated command passes --harness without --exact, so a selector can match more than the manifest lists"
    )]
    NotExact,
    /// The same harness is selected twice.
    #[error("the delegated command selects these harnesses more than once: {names:?}")]
    DuplicateSelector {
        /// The repeated selector names.
        names: Vec<String>,
    },
}

/// Split leading `NAME=VALUE` words from the command words that follow.
///
/// A value wrapped in one pair of double quotes loses them, because a Make
/// recipe's `KANI_ENV` carries the quotes a shell assignment would have
/// removed and the checker receives the word verbatim.
fn split_assignments(words: &[String]) -> (Vec<(String, String)>, &[String]) {
    let mut env = Vec::new();
    let mut index = 0;
    while let Some(word) = words.get(index) {
        let Some(assignment) = assignment_of(word) else {
            break;
        };
        env.push(assignment);
        index += 1;
    }
    (env, words.get(index..).unwrap_or(&[]))
}

/// Return the environment assignment `word` spells, if it is one.
fn assignment_of(word: &str) -> Option<(String, String)> {
    let (name, value) = word.split_once('=')?;
    if !is_variable_name(name) {
        return None;
    }
    Some((name.to_owned(), unquoted(value)))
}

/// Return whether `name` can be an environment variable name.
///
/// The delegated command's leading assignments reach a child process, so a
/// word that is only shaped like one is refused rather than passed through.
fn is_variable_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// Return `value` without one surrounding pair of double quotes.
fn unquoted(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(value)
        .to_owned()
}

/// Validate a delegated command against the package and manifest it must use.
///
/// # Errors
///
/// Returns [`CommandError`] when the command is not a cargo-kani invocation,
/// passes an option that hides or redirects its output, selects a different
/// package, omits `--lib`, or selects harnesses without `--exact`.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::kani_suite::runner::parse_command;
///
/// let words = ["LD_LIBRARY_PATH=/tmp/lib", "cargo", "kani", "-p", "pkg",
///              "--lib", "--exact", "--harness", "proofs::verify_one"]
///     .map(str::to_owned);
/// let delegated =
///     parse_command(&words, "pkg").expect("a valid command");
/// assert_eq!(delegated.env, vec![("LD_LIBRARY_PATH".to_owned(), "/tmp/lib".to_owned())]);
/// assert_eq!(delegated.selectors, vec!["proofs::verify_one"]);
/// ```
pub fn parse_command(words: &[String], package: &str) -> Result<Delegated, CommandError> {
    let (env, command) = split_assignments(words);
    let Some((program, args)) = command.split_first() else {
        return Err(CommandError::Empty);
    };
    let subcommand = cargo_options::subcommand_index(args);
    if args.get(subcommand).map(String::as_str) != Some("kani") {
        return Err(CommandError::NotCargoKani {
            args: args.to_vec(),
        });
    }
    let mut parser = parser::Parser::default();
    let mut index = subcommand + 1;
    while index < args.len() {
        index = parser.consume(args, index)?;
    }
    let invocation = Invocation {
        env,
        program: program.clone(),
        args: args.to_vec(),
    };
    parser.finish(&invocation, package)
}

/// The words that will be executed, before validation decides whether to.
///
/// Grouped rather than threaded through [`Parser::finish`] as three
/// parameters: they travel together and are consumed together, and a
/// four-parameter ceiling keeps a reader from having to hold five positional
/// arguments in mind.
pub(super) struct Invocation {
    /// Environment assignments applied to the child alone.
    pub(super) env: Vec<(String, String)>,
    /// The program to run.
    pub(super) program: String,
    /// The program's arguments.
    pub(super) args: Vec<String>,
}

#[cfg(test)]
mod tests {
    //! Unit tests for the delegated command's argument positions.

    use rstest::rstest;

    use super::*;

    /// Return a parameterised case's words as owned arguments.
    fn owned(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    #[test]
    fn a_plain_cargo_invocation_still_resolves_kani() {
        let words = ["cargo", "kani", "-p", "pkg", "--lib", "--exact"];
        let delegated =
            parse_command(&owned(&words), "pkg").expect("a plain cargo kani invocation");
        assert_eq!(delegated.program, "cargo");
    }

    #[rstest]
    #[case(
        &["cargo", "kani", "-p", "pkg", "other", "--lib", "--exact"],
        "a second package"
    )]
    #[case(
        &["cargo", "kani", "-p", "pkg", "--lib", "--exact", "--harness", "a"],
        "one package"
    )]
    fn only_one_package_may_be_selected(#[case] words: &[&str], #[case] what: &str) {
        let outcome = parse_command(&owned(words), "pkg");
        match what {
            "one package" => assert!(outcome.is_ok(), "one package must be accepted"),
            _ => assert!(
                matches!(outcome, Err(CommandError::Package { .. })),
                "a second package must be rejected, got {outcome:?}"
            ),
        }
    }

    #[test]
    fn a_compact_package_option_selects_its_package() {
        let words = ["cargo", "kani", "-ppkg", "--lib", "--exact"];
        let delegated =
            parse_command(&owned(&words), "pkg").expect("a compact -p selects the package");
        assert_eq!(delegated.package, "pkg");
    }

    #[test]
    fn a_compact_package_option_still_refuses_a_second_package() {
        let words = ["cargo", "kani", "-ppkg", "other", "--lib", "--exact"];
        let outcome = parse_command(&owned(&words), "pkg");
        assert!(
            matches!(outcome, Err(CommandError::Package { .. })),
            "a compact -p followed by a second package must be rejected, got {outcome:?}"
        );
    }

    #[rstest]
    #[case(
        &["cargo", "kani", "-p", "pkg", "--lib", "--bin", "helper"],
        "a bin option"
    )]
    #[case(
        &["cargo", "kani", "-p", "pkg", "--lib", "--bin=helper"],
        "an attached bin option"
    )]
    #[case(&["cargo", "kani", "-p", "pkg", "--lib", "--bins"], "the bins flag")]
    fn a_binary_target_is_refused(#[case] words: &[&str], #[case] what: &str) {
        let outcome = parse_command(&owned(words), "pkg");
        assert!(
            matches!(
                outcome,
                Err(CommandError::BinaryTarget { .. } | CommandError::AllBinaries)
            ),
            "{what} must be refused, got {outcome:?}"
        );
    }
}
