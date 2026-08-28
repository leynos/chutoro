//! Small helpers shared across CLI tests.
//!
//! The CLI unit tests build temporary input files and assert error handling
//! behaviour. These helpers keep the test cases concise and consistent.
//!
//! Setup helpers here are fallible: they surface failures to the calling test
//! rather than panicking on its behalf, so diagnostics point at the test that
//! actually failed.

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

use rstest::fixture;
use tempfile::TempDir;

use super::super::{Cli, Command, RunCommand, RunSource, TextArgs, TextMetric};

/// Provides a scratch directory for tests that write input files.
///
/// The fixture is fallible so the temporary-directory failure is reported by
/// the consuming test rather than by this helper.
#[fixture]
pub(super) fn temp_dir() -> io::Result<TempDir> {
    TempDir::new()
}

pub(super) fn create_text_file(dir: &TempDir, name: &str, contents: &str) -> io::Result<PathBuf> {
    let path = dir.path().join(name);
    let mut file = File::create(&path)?;
    file.write_all(contents.as_bytes())?;
    Ok(path)
}

/// Builds a text-source `run` command for `path`.
///
/// Every text CLI test shares the same Levenshtein metric and derived data
/// source name, so only the tuning knobs vary between call sites.
pub(super) fn text_run_command(
    path: PathBuf,
    min_cluster_size: usize,
    max_bytes: Option<u64>,
) -> RunCommand {
    RunCommand {
        min_cluster_size,
        max_bytes,
        source: RunSource::Text(TextArgs {
            path,
            metric: TextMetric::Levenshtein,
            name: None,
        }),
    }
}

/// Wraps [`text_run_command`] in a parsed [`Cli`] invocation.
pub(super) fn text_cli(path: PathBuf, min_cluster_size: usize, max_bytes: Option<u64>) -> Cli {
    Cli {
        command: Command::Run(text_run_command(path, min_cluster_size, max_bytes)),
    }
}

/// Asserts that `$expr` evaluates to `Err` and yields the wrapped error.
///
/// Implemented as a macro (rather than a helper function) so a failed
/// assertion's panic location points at the calling test, matching the
/// diagnostics produced by the `expect_err`-style helpers it replaces.
macro_rules! expect_err {
    ($expr:expr, $($panic_arg:tt)+) => {
        match $expr {
            Err(error) => error,
            Ok(_) => panic!($($panic_arg)+),
        }
    };
}

pub(super) use expect_err;
