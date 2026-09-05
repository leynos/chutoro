//! Locate compiled `[[bin]]` target binaries for behavioural tests that spawn
//! CLI subprocesses.
//!
//! This module centralizes the binary-resolution logic that Cargo
//! integration tests otherwise reimplement per binary: prefer the
//! `CARGO_BIN_EXE_<name>` environment variable Cargo sets for binary targets
//! owned by the crate under test, and fall back to probing the target
//! directory when that variable is unavailable. [`find_test_binary`] is the
//! sole public entry point; callers pass the binary's `[[bin]]` `name` and
//! receive its resolved path.

use cap_std::{
    ambient_authority,
    fs::{Dir, DirEntry},
};
use std::env;
use std::path::{Path, PathBuf};
use thiserror::Error;

use mockable::{DefaultEnv, Env};

/// Errors surfaced when a compiled test binary cannot be located.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TestBinaryError {
    /// The current test binary's own executable path could not be resolved.
    #[error("failed to locate current test binary: {0}")]
    CurrentExe(String),
    /// The `deps` directory could not be derived from the test binary path.
    #[error("failed to resolve deps directory from test binary")]
    DepsDir,
    /// The `target` directory could not be derived from the `deps` directory.
    #[error("failed to resolve target directory from deps")]
    TargetDir,
    /// No binary matching the requested name was found under the target
    /// directory.
    #[error("failed to locate {name} binary")]
    NotFound {
        /// Name of the binary that could not be located.
        name: String,
    },
}

/// Locates a compiled binary produced by a `[[bin]]` target named `name` in
/// the crate under test.
///
/// Integration tests that spawn a sibling binary as a subprocess (for
/// example, a CI gate binary) need its path. Cargo sets
/// `CARGO_BIN_EXE_<name>` for exactly this purpose when the binary target
/// belongs to the same package as the test; this function prefers that
/// variable and only falls back to probing `target/<profile>/` and its
/// `deps/` subdirectory when the variable is absent.
///
/// # Errors
///
/// Returns [`TestBinaryError`] when the current test binary's path, its
/// `deps` directory, or the `target` directory cannot be resolved, or when no
/// binary named `name` is found.
///
/// # Examples
///
/// ```no_run
/// use chutoro_test_support::process::find_test_binary;
///
/// let path = find_test_binary("kani_nightly_gate").expect("binary must exist");
/// assert!(path.exists());
/// ```
pub fn find_test_binary(name: &str) -> Result<PathBuf, TestBinaryError> {
    find_test_binary_with_env(name, &DefaultEnv)
}

/// Resolve a binary through an injected environment reader before probing disk.
fn find_test_binary_with_env(name: &str, env: &dyn Env) -> Result<PathBuf, TestBinaryError> {
    if let Some(path) = env.path_buf(&format!("CARGO_BIN_EXE_{name}")) {
        return Ok(with_exe_suffix(path));
    }

    let current_exe =
        env::current_exe().map_err(|error| TestBinaryError::CurrentExe(error.to_string()))?;
    resolve_test_binary_from_executable(name, Ok(current_exe))
}

/// Resolve a named binary from the supplied current executable's target layout.
fn resolve_test_binary_from_executable(
    name: &str,
    current_exe: Result<PathBuf, TestBinaryError>,
) -> Result<PathBuf, TestBinaryError> {
    let executable_path = current_exe?;
    let deps_dir = executable_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or(TestBinaryError::DepsDir)?;
    let target_dir = deps_dir
        .parent()
        .map(Path::to_path_buf)
        .ok_or(TestBinaryError::TargetDir)?;
    let direct = with_exe_suffix(target_dir.join(name));
    if direct.exists() {
        return Ok(direct);
    }

    find_in_deps(&deps_dir, name).ok_or_else(|| TestBinaryError::NotFound {
        name: name.to_owned(),
    })
}

/// Resolves a test binary from an optional environment path or executable path.
#[cfg(test)]
fn find_test_binary_with(
    name: &str,
    environment_path: Option<PathBuf>,
    current_exe: Result<PathBuf, TestBinaryError>,
) -> Result<PathBuf, TestBinaryError> {
    if let Some(path) = environment_path {
        return Ok(with_exe_suffix(path));
    }

    resolve_test_binary_from_executable(name, current_exe)
}

/// Search the compiled dependency directory for a binary matching `name`.
fn find_in_deps(deps_dir: &Path, name: &str) -> Option<PathBuf> {
    Dir::open_ambient_dir(deps_dir, ambient_authority())
        .ok()?
        .entries()
        .ok()?
        .filter_map(Result::ok)
        .find_map(|entry| is_matching_binary(&entry, deps_dir, name))
}

/// Return the entry's path when it is a compiled binary matching `name`.
fn is_matching_binary(entry: &DirEntry, deps_dir: &Path, name: &str) -> Option<PathBuf> {
    let os_file_name = entry.file_name();
    let file_name = os_file_name.to_str()?;
    let path = deps_dir.join(file_name);
    let metadata = entry.metadata().ok()?;

    if !metadata.is_file() {
        return None;
    }

    if !has_expected_suffix(&path, file_name) {
        return None;
    }

    let file_stem = path.file_stem()?.to_str()?;
    let hyphenated_prefix = format!("{name}-");
    (file_stem == name || file_stem.starts_with(&hyphenated_prefix)).then_some(path)
}

/// Report whether a candidate binary has the platform's executable suffix.
fn has_expected_suffix(path: &Path, file_name: &str) -> bool {
    let suffix = env::consts::EXE_SUFFIX;
    if suffix.is_empty() {
        return path.extension().is_none();
    }

    file_name.ends_with(suffix)
}

/// Append the platform executable suffix unless `path` already has it.
fn with_exe_suffix(mut path: PathBuf) -> PathBuf {
    let suffix = env::consts::EXE_SUFFIX;
    if suffix.is_empty() {
        return path;
    }

    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return path;
    };
    if file_name.ends_with(suffix) {
        return path;
    }

    let updated = format!("{file_name}{suffix}");
    path.set_file_name(updated);
    path
}

#[cfg(test)]
mod tests {
    //! Tests for test-binary resolution paths and errors.

    use super::*;
    use mockable::MockEnv;

    #[test]
    fn resolves_binary_from_environment_path() {
        let mut env = MockEnv::new();
        env.expect_path_buf().returning(|key| {
            assert_eq!(key, "CARGO_BIN_EXE_test-binary");
            Some(PathBuf::from("test-binary"))
        });
        let path = find_test_binary_with_env("test-binary", &env)
            .expect("environment path should resolve the test binary");

        assert_eq!(path, with_exe_suffix(PathBuf::from("test-binary")));
    }

    #[test]
    fn resolves_binary_from_environment_fallback() {
        let mut env = MockEnv::new();
        env.expect_path_buf().returning(|key| {
            assert_eq!(key, "CARGO_BIN_EXE_chutoro_test_support");
            None
        });

        let path = find_test_binary_with_env("chutoro_test_support", &env)
            .expect("fallback should resolve this test binary");

        assert!(path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn resolves_non_unicode_binary_from_environment_path() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        let expected = PathBuf::from(OsString::from_vec(b"test-binary-\xFF".to_vec()));
        let expected_for_mock = expected.clone();
        let mut env = MockEnv::new();
        env.expect_path_buf().returning(move |key| {
            assert_eq!(key, "CARGO_BIN_EXE_test-binary");
            Some(expected_for_mock.clone())
        });

        let path = find_test_binary_with_env("test-binary", &env)
            .expect("non-Unicode environment path should resolve the test binary");

        assert_eq!(path, expected);
    }

    #[test]
    fn resolves_binary_from_dependency_directory_fallback() {
        let current_exe =
            env::current_exe().map_err(|error| TestBinaryError::CurrentExe(error.to_string()));
        let path = find_test_binary_with("chutoro_test_support", None, current_exe)
            .expect("dependency-directory fallback should resolve this test binary");

        assert!(path.exists());
    }

    #[test]
    fn rejects_non_executable_suffix() {
        let valid_name = format!("test-binary{}", env::consts::EXE_SUFFIX);
        assert!(has_expected_suffix(Path::new(&valid_name), &valid_name));
        assert!(!has_expected_suffix(
            Path::new("test-binary.invalid"),
            "test-binary.invalid"
        ));
    }

    #[test]
    fn returns_not_found_when_no_candidate_exists() {
        let error = find_test_binary_with(
            "missing-test-binary",
            None,
            Ok(PathBuf::from(
                "/chutoro-test-support-missing/target/debug/deps/test-binary",
            )),
        )
        .expect_err("missing binary should produce a NotFound error");

        assert_eq!(
            error,
            TestBinaryError::NotFound {
                name: "missing-test-binary".to_owned(),
            }
        );
    }
}
