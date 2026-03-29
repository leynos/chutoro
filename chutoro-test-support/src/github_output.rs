//! Capability-based helpers for writing GitHub Actions command files.
//!
//! GitHub Actions exposes paths such as `GITHUB_OUTPUT` through environment
//! variables. These helpers keep the ambient authority boundary narrow by
//! opening only the parent directory and then appending to the named file
//! relative to that directory.

use std::env;
use std::ffi::OsStr;
use std::io::{self, Write};
use std::path::Path;

use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};

const GITHUB_OUTPUT_DELIMITER: &str = "CHUTORO_EOF";

/// A single key/value pair to append to a GitHub Actions command file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputPair<'a> {
    key: &'a str,
    value: &'a str,
}

impl<'a> OutputPair<'a> {
    /// Creates a key/value pair for `GITHUB_OUTPUT`.
    ///
    /// # Examples
    /// ```rust
    /// use chutoro_test_support::github_output::OutputPair;
    ///
    /// let pair = OutputPair::new("mode", "baseline_compare");
    /// assert_eq!(pair.key(), "mode");
    /// assert_eq!(pair.value(), "baseline_compare");
    /// ```
    #[must_use]
    pub fn new(key: &'a str, value: &'a str) -> Self {
        Self { key, value }
    }

    /// Returns the key written to the command file.
    ///
    /// # Examples
    /// ```rust
    /// use chutoro_test_support::github_output::OutputPair;
    ///
    /// let pair = OutputPair::new("should_run", "true");
    /// assert_eq!(pair.key(), "should_run");
    /// ```
    #[must_use]
    pub fn key(self) -> &'a str {
        self.key
    }

    /// Returns the value written to the command file.
    ///
    /// # Examples
    /// ```rust
    /// use chutoro_test_support::github_output::OutputPair;
    ///
    /// let pair = OutputPair::new("reason", "scheduled baseline");
    /// assert_eq!(pair.value(), "scheduled baseline");
    /// ```
    #[must_use]
    pub fn value(self) -> &'a str {
        self.value
    }
}

/// Appends entries to the `GITHUB_OUTPUT` command file when it is configured.
///
/// When `GITHUB_OUTPUT` is unset or empty this is a no-op so local runs behave
/// like GitHub Actions command files being disabled.
///
/// # Examples
/// ```no_run
/// use chutoro_test_support::github_output::{OutputPair, append_if_configured};
///
/// append_if_configured(&[
///     OutputPair::new("mode", "baseline_compare"),
///     OutputPair::new("should_compare", "true"),
/// ])?;
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn append_if_configured(entries: &[OutputPair<'_>]) -> io::Result<()> {
    let Some(output_path) = read_optional_env("GITHUB_OUTPUT")? else {
        return Ok(());
    };
    if output_path.is_empty() {
        return Ok(());
    }

    let mut file = open_output_file(Path::new(&output_path))?;
    for entry in entries {
        write_github_output_value(&mut file, entry.key(), entry.value())?;
    }

    Ok(())
}

fn read_optional_env(name: &str) -> io::Result<Option<String>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(io::Error::other(error)),
    }
}

fn open_output_file(output_path: &Path) -> io::Result<cap_std::fs::File> {
    let (parent, file_name) = split_output_path(output_path)?;
    let dir = Dir::open_ambient_dir(parent, ambient_authority())?;
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    dir.open_with(file_name, &options)
}

fn split_output_path(output_path: &Path) -> io::Result<(&Path, &OsStr)> {
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let Some(file_name) = output_path.file_name() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid GITHUB_OUTPUT path: {}", output_path.display()),
        ));
    };

    Ok((parent, file_name))
}

fn write_github_output_value(file: &mut impl Write, key: &str, value: &str) -> io::Result<()> {
    if !value.contains('\n') && !value.contains('\r') {
        writeln!(file, "{key}={value}")?;
        return Ok(());
    }

    if value.contains(GITHUB_OUTPUT_DELIMITER) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("output value for {key} contains {GITHUB_OUTPUT_DELIMITER}"),
        ));
    }

    writeln!(file, "{key}<<{GITHUB_OUTPUT_DELIMITER}")?;
    writeln!(file, "{value}")?;
    writeln!(file, "{GITHUB_OUTPUT_DELIMITER}")?;

    Ok(())
}
