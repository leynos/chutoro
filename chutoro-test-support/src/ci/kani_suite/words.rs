//! Interpret one delegated `cargo kani` argument.
//!
//! [`super::parser`] walks the delegated argument list and keeps the run's
//! state; the questions about a single argument live here: whether it spells a
//! refused option, which value-taking option it names, and which selectors
//! repeat. Keeping the spellings in one module is what stops the separated and
//! attached forms of an option from disagreeing about it.

use std::collections::BTreeMap;

use super::parser::Parser;
use super::runner::CommandError;

/// An option that takes its value in the following argument.
///
/// The four here are the options with a separated spelling: the output format,
/// which is checked rather than stored, and the package and harness selectors
/// and the binary target, which are recorded. Every `store` returns a `Result`
/// even where recording cannot fail, so the checked option can sit beside the
/// others without a second code path.
#[derive(Debug, Clone, Copy)]
pub(super) enum Valued {
    /// The output format, which must be the regular one.
    OutputFormat,
    /// A package selector.
    Package,
    /// A harness selector.
    Harness,
    /// A binary target, which must not be selected at all.
    Binary,
}

impl Valued {
    /// Return the value-taking option `argument` names, if it is one.
    pub(super) fn of(argument: &str) -> Option<Self> {
        match argument {
            "--output-format" => Some(Self::OutputFormat),
            "-p" | "--package" => Some(Self::Package),
            "--harness" => Some(Self::Harness),
            "--bin" => Some(Self::Binary),
            _ => None,
        }
    }

    /// Return whether this option accepts several values in a row.
    ///
    /// Cargo's package option takes a list; the others take exactly one value.
    pub(super) const fn repeats(self) -> bool {
        matches!(self, Self::Package)
    }

    /// Store or check `value` for the option this value names.
    pub(super) fn store(self, parser: &mut Parser, value: String) -> Result<(), CommandError> {
        match self {
            Self::OutputFormat => check_output_format(&value),
            Self::Package => {
                parser.push_package(value);
                Ok(())
            }
            Self::Harness => {
                parser.push_selector(value);
                Ok(())
            }
            Self::Binary => Err(CommandError::BinaryTarget { name: value }),
        }
    }
}

/// Report whether `argument` spells the refused option `flag`.
///
/// `--jobs 4`, `--jobs=4` and `-j4` are one option in three spellings, and a
/// short option takes its value attached without a separator. Matching only
/// the exact flag and its `=` form would let `-j4` reach the child and
/// interleave the blocks the checker reads.
pub(super) fn matches_forbidden(argument: &str, flag: &str) -> bool {
    if argument == flag || argument.starts_with(&format!("{flag}=")) {
        return true;
    }
    // A short option is one dash and one letter, so anything longer that
    // starts with it is the option with its value attached.
    flag.len() == 2
        && flag.starts_with('-')
        && !flag.starts_with("--")
        && argument.starts_with(flag)
        && argument.len() > flag.len()
}

/// Reject any output format other than the regular one.
pub(super) fn check_output_format(format: &str) -> Result<(), CommandError> {
    if format == "regular" {
        return Ok(());
    }
    Err(CommandError::OutputFormat {
        format: format.to_owned(),
    })
}

/// Return the names appearing more than once in `names`, sorted.
pub(super) fn repeated(names: &[String]) -> Vec<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for name in names {
        *counts.entry(name.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(name, _)| name.to_owned())
        .collect()
}

/// Return the index of the subcommand in `args`, after Cargo's global options.
///
/// Cargo accepts both a toolchain selector (`+nightly`) and its global options
/// before the subcommand: `cargo --config net.offline=true kani` is a valid
/// invocation, and the repository exposes `CARGO` as a command-line override so
/// a caller can spell it that way. Both are skipped so this check is about the
/// subcommand rather than about how the toolchain or configuration is spelled.
/// An option this does not recognise is left in place, so an unexpected word
/// fails as a wrong subcommand rather than being silently ignored.
pub(super) fn subcommand_index(args: &[String]) -> usize {
    let mut index = 0;
    while let Some(word) = args.get(index) {
        if word.starts_with('+') {
            index += 1;
            continue;
        }
        let Some(consumed) = cargo_global_option(word) else {
            break;
        };
        index += consumed;
    }
    index
}

/// Return how many words the Cargo global option at `word` occupies.
///
/// A value-taking option spelled as one word (`--config=x`) occupies one; the
/// separated spelling occupies two. A flag occupies one. None means the word is
/// not a global option, so the caller has reached the subcommand.
fn cargo_global_option(word: &str) -> Option<usize> {
    const VALUED: &[&str] = &[
        "--color",
        "--config",
        "--explain",
        "--manifest-path",
        "--target-dir",
        "-C",
        "-Z",
    ];
    // Quiet is deliberately absent: this checker refuses a run whose output
    // is suppressed, and a quiet word before the subcommand is no exception.
    // Leaving it unrecognised makes the invocation fail as a wrong
    // subcommand, which is the safe direction.
    const FLAGS: &[&str] = &["--frozen", "--locked", "--offline", "--verbose", "-v"];
    if FLAGS.contains(&word) {
        return Some(1);
    }
    if VALUED.contains(&word) {
        return Some(2);
    }
    let (name, _) = word.split_once('=')?;
    VALUED.contains(&name).then_some(1)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the Cargo words this module interprets.

    use rstest::rstest;

    use super::*;

    /// Return a parameterised case's words as owned arguments.
    fn owned(args: &[&str]) -> Vec<String> {
        args.iter().map(|word| (*word).to_owned()).collect()
    }

    #[rstest]
    #[case(&["kani"], 0, "a plain command")]
    #[case(&["+nightly", "kani"], 1, "a toolchain selector")]
    #[case(&["--config", "net.offline=true", "kani"], 2, "a separated option")]
    #[case(&["--config=net.offline=true", "kani"], 1, "an attached option")]
    #[case(&["--locked", "+nightly", "kani"], 2, "a flag before the selector")]
    fn the_subcommand_follows_cargo_global_options(
        #[case] args: &[&str],
        #[case] expected: usize,
        #[case] what: &str,
    ) {
        let words = owned(args);
        assert_eq!(
            subcommand_index(&words),
            expected,
            "{what} must resolve to its subcommand"
        );
    }

    #[rstest]
    #[case("--config", Some(2), "a separated value")]
    #[case("--config=net.offline=true", Some(1), "an attached value")]
    #[case("--locked", Some(1), "a flag")]
    #[case("kani", None, "the subcommand")]
    fn only_cargo_global_options_are_consumed(
        #[case] word: &str,
        #[case] expected: Option<usize>,
        #[case] what: &str,
    ) {
        assert_eq!(cargo_global_option(word), expected, "{what}");
    }
}
