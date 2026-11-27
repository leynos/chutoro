//! Read Cargo's own global options before its subcommand.
//!
//! Cargo accepts a toolchain selector and its global options in front of the
//! subcommand: `cargo --config net.offline=true kani` is a valid invocation,
//! and the repository exposes `CARGO` as a command-line override so a caller
//! can spell it that way. The delegated command's validator has to find the
//! subcommand through those words without mistaking one for it.

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
}
