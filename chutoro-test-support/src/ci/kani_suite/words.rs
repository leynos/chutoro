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

    /// Store or check `value` for the option this value names.
    pub(super) fn store(self, parser: &mut Parser, value: String) -> Result<(), CommandError> {
        match self {
            Self::OutputFormat => check_output_format(&value),
            Self::Package => {
                parser.packages.push(value);
                Ok(())
            }
            Self::Harness => {
                parser.selectors.push(value);
                Ok(())
            }
            Self::Binary => Err(CommandError::BinaryTarget { name: value }),
        }
    }
}

/// Return the package a compact `-p` option selects, if `argument` is one.
///
/// Cargo's short option takes its value with no separator, so `-pkg` is the
/// same option as `-p pkg`. Reading only the separated and `=` forms would
/// leave it unrecorded, and a command that selected a package would then look
/// like one that selected none -- or worse, its second package would be read
/// as another word entirely.
pub(super) fn compact_package(argument: &str) -> Option<&str> {
    argument
        .strip_prefix("-p")
        .filter(|value| !value.is_empty() && !value.starts_with('='))
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
