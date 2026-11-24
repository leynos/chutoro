//! Read the resolved Criterion version from the workspace lockfile.

use std::path::{Path, PathBuf};

/// Find the lockfile that governs `workspace`.
///
/// A workspace member has no lockfile of its own, so the search walks up to
/// the directory that owns one. This keeps `--workspace` usable for either a
/// package directory or the workspace root.
pub(super) fn lockfile_path(workspace: &Path) -> PathBuf {
    let mut candidate = workspace.to_path_buf();
    loop {
        let lockfile = candidate.join("Cargo.lock");
        if lockfile.is_file() {
            return lockfile;
        }
        let Some(parent) = candidate.parent() else {
            return workspace.join("Cargo.lock");
        };
        candidate = parent.to_path_buf();
    }
}

/// Return the version a Cargo lockfile records for `package`.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::output_root::lockfile_package_version;
///
/// let lockfile = r#"
/// [[package]]
/// name = "criterion"
/// version = "0.5.1"
/// "#;
/// assert_eq!(lockfile_package_version(lockfile, "criterion"), Some("0.5.1".to_owned()));
/// assert_eq!(lockfile_package_version(lockfile, "tempfile"), None);
/// ```
#[must_use]
pub fn lockfile_package_version(lockfile: &str, package: &str) -> Option<String> {
    let name_line = format!("name = \"{package}\"");
    let mut lines = lockfile.lines();
    while let Some(line) = lines.next() {
        if line.trim() != name_line {
            continue;
        }
        let version_line = lines.next()?;
        return version_line
            .trim()
            .strip_prefix("version = \"")
            .and_then(|rest| rest.strip_suffix('"'))
            .map(str::to_owned);
    }

    None
}
