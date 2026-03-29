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

#[cfg(test)]
mod tests {
    //! Regression coverage for GitHub Actions output-file encoding.

    use super::{
        GITHUB_OUTPUT_DELIMITER, OutputPair, append_if_configured, split_output_path,
        write_github_output_value,
    };
    use cap_std::{ambient_authority, fs::Dir};
    use std::{
        env,
        io::{self, ErrorKind, Read},
        path::{Path, PathBuf},
        sync::{Mutex, OnceLock},
    };
    use tempfile::tempdir;

    static ENV_GUARD: OnceLock<Mutex<()>> = OnceLock::new();

    fn with_github_output_env(
        path: Option<&Path>,
        test: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        let _guard = ENV_GUARD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("test environment guard must lock");
        let previous = env::var_os("GITHUB_OUTPUT");
        match path {
            Some(path) => {
                // SAFETY: Tests serialize environment mutations behind
                // `ENV_GUARD`, so no concurrent readers or writers exist.
                unsafe { env::set_var("GITHUB_OUTPUT", path) };
            }
            None => {
                // SAFETY: Tests serialize environment mutations behind
                // `ENV_GUARD`, so no concurrent readers or writers exist.
                unsafe { env::remove_var("GITHUB_OUTPUT") };
            }
        }

        let result = test();

        match previous {
            Some(value) => {
                // SAFETY: Tests serialize environment mutations behind
                // `ENV_GUARD`, so no concurrent readers or writers exist.
                unsafe { env::set_var("GITHUB_OUTPUT", value) };
            }
            None => {
                // SAFETY: Tests serialize environment mutations behind
                // `ENV_GUARD`, so no concurrent readers or writers exist.
                unsafe { env::remove_var("GITHUB_OUTPUT") };
            }
        }

        result
    }

    fn read_github_output(path: &Path) -> io::Result<String> {
        let dir_path = path
            .parent()
            .expect("temp output path must have a parent directory");
        let file_name = path
            .file_name()
            .expect("temp output path must have a file name");
        let dir = Dir::open_ambient_dir(dir_path, ambient_authority())?;
        let mut file = dir.open(file_name)?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        Ok(buffer)
    }

    #[test]
    fn append_if_configured_unset_noop() {
        let temp_dir = tempdir().expect("tempdir must be created");
        let output_path = temp_dir.path().join("github_output.txt");

        with_github_output_env(None, || {
            append_if_configured(&[OutputPair::new("mode", "baseline_compare")])
        })
        .expect("unset GITHUB_OUTPUT must be a no-op");

        assert!(
            !output_path.exists(),
            "unset GITHUB_OUTPUT must not create an output file"
        );
    }

    #[test]
    fn append_if_configured_empty_noop() {
        let temp_dir = tempdir().expect("tempdir must be created");
        let output_path = temp_dir.path().join("github_output.txt");

        with_github_output_env(Some(Path::new("")), || {
            append_if_configured(&[OutputPair::new("mode", "baseline_compare")])
        })
        .expect("empty GITHUB_OUTPUT must be a no-op");

        assert!(
            !output_path.exists(),
            "empty GITHUB_OUTPUT must not create an output file"
        );
    }

    #[test]
    fn writes_simple_key_value() {
        let temp_dir = tempdir().expect("tempdir must be created");
        let output_path = temp_dir.path().join("github_output.txt");

        with_github_output_env(Some(&output_path), || {
            append_if_configured(&[OutputPair::new("mode", "baseline_compare")])
        })
        .expect("single-line output must write successfully");

        let contents = read_github_output(&output_path).expect("output file must be readable");
        assert_eq!(contents, "mode=baseline_compare\n");
    }

    #[test]
    fn writes_multiline_value_with_delimiter() {
        let temp_dir = tempdir().expect("tempdir must be created");
        let output_path = temp_dir.path().join("github_output.txt");
        let multiline = "line1\nline2\nline3";

        with_github_output_env(Some(&output_path), || {
            append_if_configured(&[OutputPair::new("summary", multiline)])
        })
        .expect("multiline output must write successfully");

        let contents = read_github_output(&output_path).expect("output file must be readable");
        let expected = format!(
            "summary<<{delimiter}\n{multiline}\n{delimiter}\n",
            delimiter = GITHUB_OUTPUT_DELIMITER,
        );
        assert_eq!(contents, expected);
    }

    #[test]
    fn rejects_values_containing_delimiter() {
        let temp_dir = tempdir().expect("tempdir must be created");
        let output_path = temp_dir.path().join("github_output.txt");
        let invalid = format!("line1\ncontains {GITHUB_OUTPUT_DELIMITER}");

        let err = with_github_output_env(Some(&output_path), || {
            append_if_configured(&[OutputPair::new("bad", &invalid)])
        })
        .expect_err("delimiter collision must be rejected");

        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn split_output_path_uses_parent_directory_and_file_name() {
        let output_path = PathBuf::from("nested/github_output.txt");
        let (parent, file_name) =
            split_output_path(&output_path).expect("path splitting must succeed");

        assert_eq!(parent, Path::new("nested"));
        assert_eq!(file_name, std::ffi::OsStr::new("github_output.txt"));
    }

    #[test]
    fn write_github_output_value_rejects_delimiter() {
        let mut buffer = Vec::new();
        let invalid = format!("line1\ncontains {GITHUB_OUTPUT_DELIMITER}");

        let err = write_github_output_value(&mut buffer, "bad", &invalid)
            .expect_err("delimiter collision must be rejected");

        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }
}
