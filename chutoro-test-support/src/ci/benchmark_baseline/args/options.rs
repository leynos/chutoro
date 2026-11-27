//! Parse `--name value` option pairs.
//!
//! The parser is separate from the command types so the module that declares
//! the argument structs stays a data description rather than a mix of data and
//! a scanning loop.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::super::BaselineError;
use super::usage;

/// Collected `--name value` options with repeatable values.
#[derive(Debug, Default)]
pub(super) struct Options {
    /// Option values, keyed by their long name.
    values: BTreeMap<String, Vec<String>>,
}

impl Options {
    /// Parse `--name value` pairs, rejecting unknown or misplaced arguments.
    pub(super) fn parse(arguments: &[String]) -> Result<Self, BaselineError> {
        let mut options = Self::default();
        let mut remaining = arguments.iter();
        while let Some(argument) = remaining.next() {
            let name = if Self::is_known(argument) {
                argument.clone()
            } else if argument.starts_with('-') {
                return Err(usage(&format!("unknown option `{argument}`")));
            } else {
                return Err(usage(&format!(
                    "unexpected positional argument `{argument}`"
                )));
            };
            let value = remaining
                .next()
                .ok_or_else(|| usage(&format!("option `{name}` requires a value")))?;
            options.values.entry(name).or_default().push(value.clone());
        }

        Ok(options)
    }

    /// Return whether `name` is an option this helper understands.
    pub(super) fn is_known(name: &str) -> bool {
        matches!(
            name,
            "--workspace"
                | "--bench"
                | "--baseline"
                | "--revision"
                | "--output"
                | "--fresh-since"
                | "--bench-argument"
                | "--reference-manifest"
                | "--current-manifest"
                | "--evidence"
                | "--filter-output"
                | "--preflight"
                | "--verdict"
                | "--threshold"
                | "--no-reference"
                | "--failed-stage"
                | "--failure-reason"
                | "--help"
        )
    }

    /// Return the first value supplied for `name`.
    pub(super) fn get(&self, name: &str) -> Option<&String> {
        self.values.get(name).and_then(|values| values.first())
    }

    /// Return a required option value.
    pub(super) fn required(&self, name: &str) -> Result<String, BaselineError> {
        self.get(name)
            .cloned()
            .ok_or_else(|| usage(&format!("option `{name}` is required")))
    }

    /// Return every recorded measurement argument in the order supplied.
    pub(super) fn bench_arguments(&self) -> Vec<String> {
        self.values
            .get("--bench-argument")
            .cloned()
            .unwrap_or_default()
    }

    /// Return a required option value as a path.
    pub(super) fn required_path(&self, name: &str) -> Result<PathBuf, BaselineError> {
        self.required(name).map(PathBuf::from)
    }

    /// Return an optional option value as a path.
    pub(super) fn path(&self, name: &str) -> Option<PathBuf> {
        self.get(name).map(PathBuf::from)
    }
}
