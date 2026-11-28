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

/// Report whether `argument` is a short-option cluster carrying a package.
///
/// Cargo reads `-vpkg` as `-v -p pkg`: a cluster may carry a package option
/// after other flags. Recording only the separated and `-p<value>` forms would
/// let such a word select a package the checker never records, and a second
/// package could then satisfy the expected set. An unattributable cluster is
/// therefore refused rather than read as a plain flag.
///
/// Only clusters that begin with a lowercase letter are considered. An
/// uppercase short option (`-Zpolonius`, `-Cpanic=abort`) takes an attached
/// value, which is Cargo's own spelling rather than a cluster.
pub(super) fn clustered_package(argument: &str) -> bool {
    let Some(cluster) = argument.strip_prefix('-') else {
        return false;
    };
    if cluster.starts_with('-') || cluster.starts_with('p') {
        return false;
    }
    cluster.starts_with(|character: char| character.is_ascii_lowercase()) && cluster.contains('p')
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

#[cfg(test)]
mod tests {
    //! Unit tests for the spellings one delegated argument can take.

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("--output-format", Some(Valued::OutputFormat))]
    #[case("-p", Some(Valued::Package))]
    #[case("--package", Some(Valued::Package))]
    #[case("--harness", Some(Valued::Harness))]
    #[case("--bin", Some(Valued::Binary))]
    #[case("--bin=helper", None)]
    #[case("--exact", None)]
    fn valued_options_are_recognised(#[case] argument: &str, #[case] expected: Option<Valued>) {
        assert_eq!(
            Valued::of(argument).map(|valued| format!("{valued:?}")),
            expected.map(|valued| format!("{valued:?}"))
        );
    }

    #[rstest]
    #[case("-pkg", Some("kg"))]
    #[case("-p", None)]
    #[case("-p=pkg", None)]
    #[case("--package=pkg", None)]
    #[case("-ppkg", Some("pkg"))]
    fn a_compact_package_carries_its_value(#[case] argument: &str, #[case] expected: Option<&str>) {
        assert_eq!(compact_package(argument), expected);
    }

    #[rstest]
    #[case("--quiet", "--quiet", true)]
    #[case("--quiet=1", "--quiet", true)]
    #[case("--quietly", "--quiet", false)]
    #[case("-j4", "-j", true)]
    #[case("-jobs", "-j", true)]
    #[case("-j", "-j", true)]
    #[case("--jobs=4", "--jobs", true)]
    #[case("--jobs", "--jobs", true)]
    #[case("--job", "--jobs", false)]
    #[case("-p", "-j", false)]
    fn forbidden_spellings_are_matched_by_shape(
        #[case] argument: &str,
        #[case] flag: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(matches_forbidden(argument, flag), expected);
    }

    #[rstest]
    #[case("-vpother", true)]
    #[case("-vppkg", true)]
    #[case("-qvpkg", true)]
    #[case("-pkg", false)]
    #[case("-p", false)]
    #[case("-p=pkg", false)]
    #[case("-vv", false)]
    #[case("-v", false)]
    #[case("-Zpolonius", false)]
    #[case("-Cpanic=abort", false)]
    #[case("--package=pkg", false)]
    fn only_a_cluster_with_a_package_is_refused(#[case] argument: &str, #[case] clustered: bool) {
        assert_eq!(clustered_package(argument), clustered);
    }

    #[rstest]
    #[case("regular", true)]
    #[case("json", false)]
    #[case("terse", false)]
    #[case("", false)]
    fn only_the_regular_output_format_is_accepted(#[case] format: &str, #[case] accepted: bool) {
        assert_eq!(check_output_format(format).is_ok(), accepted);
    }

    #[test]
    fn repeated_names_are_sorted_and_unique() {
        let names = [
            "b".to_owned(),
            "a".to_owned(),
            "b".to_owned(),
            "a".to_owned(),
            "c".to_owned(),
        ];
        assert_eq!(repeated(&names), vec!["a".to_owned(), "b".to_owned()]);
        assert!(
            repeated(&names[..1]).is_empty(),
            "one occurrence is not a repeat"
        );
    }
}
