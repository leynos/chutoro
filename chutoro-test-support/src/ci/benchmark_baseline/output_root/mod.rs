//! Resolve Criterion's output root exactly as the pinned release does.
//!
//! Criterion 0.5.1 consults `CRITERION_HOME`, then `CARGO_TARGET_DIR`, then the
//! `target_directory` reported by `cargo metadata`, then `./target/criterion`.
//! A relative value is taken relative to the benchmark process's working
//! directory, which for `cargo bench` is the workspace root. Resolving this
//! root from the environment alone would miss a configured target directory;
//! resolving it from `cargo metadata` alone would miss `CRITERION_HOME`.

use std::path::{Path, PathBuf};
use std::process::Command;

use mockable::Env;
use serde::Deserialize;

use super::BaselineError;

mod lockfile;

pub use lockfile::lockfile_package_version;

/// Environment variable that names Criterion's output root directly.
pub const CRITERION_HOME_ENV: &str = "CRITERION_HOME";

/// Environment variable that Cargo and Criterion use for the target root.
pub const CARGO_TARGET_DIR_ENV: &str = "CARGO_TARGET_DIR";

/// Inputs to the pure output-root resolver.
#[derive(Clone, Copy, Debug)]
pub struct OutputRootInputs<'a> {
    /// Value of `CRITERION_HOME`, when set.
    pub criterion_home: Option<&'a Path>,
    /// Value of `CARGO_TARGET_DIR`, when set.
    pub cargo_target_dir: Option<&'a Path>,
    /// `target_directory` reported by `cargo metadata`, when consulted.
    pub metadata_target_dir: Option<&'a Path>,
    /// Working directory Criterion would resolve relative values against.
    pub working_directory: &'a Path,
}

/// Resolve Criterion's output root for the supplied inputs.
///
/// `metadata_target_dir` is `None` when metadata was not consulted. The caller
/// that needs metadata must fail loudly rather than pass `None` after a
/// metadata failure, because the fallback below is Criterion's own last resort
/// for a directory that is not a Cargo workspace.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use chutoro_test_support::ci::benchmark_baseline::output_root::{
///     OutputRootInputs, resolve_output_root,
/// };
///
/// let root = resolve_output_root(&OutputRootInputs {
///     criterion_home: Some(Path::new("custom/criterion")),
///     cargo_target_dir: None,
///     metadata_target_dir: None,
///     working_directory: Path::new("/work"),
/// });
/// assert_eq!(root, Path::new("/work/custom/criterion"));
/// ```
#[must_use]
pub fn resolve_output_root(inputs: &OutputRootInputs<'_>) -> PathBuf {
    if let Some(home) = inputs.criterion_home {
        return resolve_relative(home, inputs.working_directory);
    }
    if let Some(target) = inputs.cargo_target_dir {
        return resolve_relative(&target.join("criterion"), inputs.working_directory);
    }
    if let Some(target) = inputs.metadata_target_dir {
        return target.join("criterion");
    }
    inputs.working_directory.join("target").join("criterion")
}

/// Resolve a possibly relative path against the working directory.
fn resolve_relative(path: &Path, working_directory: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        working_directory.join(path)
    }
}

/// Resolve the output root for a workspace, consulting metadata only if needed.
///
/// # Errors
///
/// Returns an error when `cargo metadata` fails or a tool version cannot be read.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::output_root::{
///     OutputRootInputs, resolve_output_root,
/// };
///
/// // The pure resolver is the unit under test; the metadata wrapper below
/// // supplies the same `target_directory` that Cargo reports.
/// let inputs = OutputRootInputs {
///     criterion_home: None,
///     cargo_target_dir: None,
///     metadata_target_dir: None,
///     working_directory: std::path::Path::new("."),
/// };
/// assert!(resolve_output_root(&inputs).ends_with("target/criterion"));
/// ```
pub fn resolve_workspace_output_root(
    workspace: &Path,
    env: &dyn Env,
) -> Result<PathBuf, BaselineError> {
    let criterion_home = path_override(env, CRITERION_HOME_ENV)?;
    // Criterion does not consult a lower-priority override once a higher one
    // is set, so neither does the helper. Reading it unconditionally would
    // reject a non-Unicode `CARGO_TARGET_DIR` that Criterion never looks at.
    let cargo_target_dir = if criterion_home.is_none() {
        path_override(env, CARGO_TARGET_DIR_ENV)?
    } else {
        None
    };
    let metadata_target_dir = if criterion_home.is_none() && cargo_target_dir.is_none() {
        Some(metadata_target_directory(workspace, env)?)
    } else {
        None
    };

    Ok(resolve_output_root(&OutputRootInputs {
        criterion_home: criterion_home.as_deref(),
        cargo_target_dir: cargo_target_dir.as_deref(),
        metadata_target_dir: metadata_target_dir.as_deref(),
        working_directory: workspace,
    }))
}

/// Read a path-valued environment override without discarding non-Unicode.
///
/// `Env::string` maps a non-Unicode value to `None`, which would silently
/// fall through to a lower-priority directory; Criterion reads the value with
/// `var_os` and would use it. Reporting the override as an error keeps the
/// helper from inventorying a directory Criterion did not write.
fn path_override(env: &dyn Env, key: &'static str) -> Result<Option<PathBuf>, BaselineError> {
    match env.raw(key) {
        Ok(value) => Ok(Some(PathBuf::from(value))),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(BaselineError::OutputRoot {
            reason: format!("{key} is set to a non-Unicode path"),
            remedy: "set the override to a UTF-8 path, or unset it so metadata is consulted",
        }),
    }
}

/// Parsed subset of `cargo metadata` output.
#[derive(Debug, Deserialize)]
pub struct CargoMetadata {
    /// Directory Cargo writes build artefacts into.
    target_directory: PathBuf,
}

impl CargoMetadata {
    /// Return Cargo's target directory.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::output_root::parse_metadata;
    ///
    /// let metadata = parse_metadata(
    ///     br#"{"target_directory":"/work/target","packages":[]}"#,
    ///     std::path::Path::new("/work"),
    /// )
    /// .expect("valid metadata");
    /// assert_eq!(metadata.target_directory(), std::path::Path::new("/work/target"));
    /// ```
    #[must_use]
    pub fn target_directory(&self) -> &Path {
        &self.target_directory
    }
}

/// Parse `cargo metadata` output.
///
/// # Errors
///
/// Returns an error when `cargo metadata` fails or a tool version cannot be read.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::output_root::parse_metadata;
///
/// assert!(parse_metadata(b"not json", std::path::Path::new(".")).is_err());
/// ```
pub fn parse_metadata(bytes: &[u8], workspace: &Path) -> Result<CargoMetadata, BaselineError> {
    serde_json::from_slice(bytes).map_err(|source| BaselineError::MetadataJson {
        workspace: workspace.to_path_buf(),
        source,
        remedy: "run `cargo metadata --format-version 1 --locked` in the workspace and fix the reported error",
    })
}

/// Return Cargo's target directory from `cargo metadata`.
///
/// # Errors
///
/// Returns an error when `cargo metadata` fails or a tool version cannot be read.
///
/// # Examples
///
/// ```
/// # use mockable::DefaultEnv;
/// # use chutoro_test_support::ci::benchmark_baseline::output_root::metadata_target_directory;
/// # let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
/// let target = metadata_target_directory(workspace, &DefaultEnv).expect("metadata must run");
/// assert!(target.is_absolute());
/// ```
pub fn metadata_target_directory(
    workspace: &Path,
    env: &dyn Env,
) -> Result<PathBuf, BaselineError> {
    let metadata = run_metadata(workspace, env)?;
    Ok(metadata.target_directory)
}

/// Return the resolved Criterion version recorded by the workspace lockfile.
///
/// The version is read from `Cargo.lock` rather than from `cargo metadata`.
/// Full metadata resolves the whole dependency graph, which needs the crate
/// cache and can block behind other builds on a shared host; the lockfile is
/// already the resolved answer for a `--locked` build and costs one file read.
fn criterion_version(workspace: &Path) -> Result<String, BaselineError> {
    let lock_path = lockfile::lockfile_path(workspace);
    let text = std::fs::read_to_string(&lock_path).map_err(|source| BaselineError::Read {
        path: lock_path.clone(),
        source,
        remedy: "run `cargo generate-lockfile` in the workspace before recording a baseline",
    })?;
    lockfile_package_version(&text, "criterion").ok_or_else(|| BaselineError::ToolVersion {
        tool: "criterion",
        message: format!(
            "no criterion package is resolved in {}",
            lock_path.display()
        ),
        remedy: "add criterion to the benchmark crate and run `cargo update -p criterion`",
    })
}

/// Tool versions recorded in every inventory manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordedTools {
    /// Resolved Criterion version.
    pub criterion: String,
    /// `cargo --version` output.
    pub cargo: String,
    /// `rustc -Vv` output.
    pub rustc: String,
}

/// Record the tool versions that produced a baseline.
///
/// # Errors
///
/// Returns an error when `cargo metadata` fails or a tool version cannot be read.
///
/// # Examples
///
/// ```
/// # use mockable::DefaultEnv;
/// # use chutoro_test_support::ci::benchmark_baseline::output_root::record_tool_versions;
/// # let crate_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
/// # let workspace = crate_directory.parent().expect("crate is in the workspace");
/// let tools = record_tool_versions(workspace, &DefaultEnv).expect("tool versions must run");
/// assert!(!tools.cargo.is_empty());
/// ```
pub fn record_tool_versions(
    workspace: &Path,
    env: &dyn Env,
) -> Result<RecordedTools, BaselineError> {
    Ok(RecordedTools {
        criterion: criterion_version(workspace)?,
        cargo: run_tool_version(workspace, env, "cargo", &["--version"])?,
        rustc: run_tool_version(workspace, env, "rustc", &["-Vv"])?,
    })
}

/// Run `cargo metadata` in the workspace and parse its output.
///
/// `--no-deps` is deliberate: only `target_directory` is needed, and full
/// metadata resolution needs the crate cache and can block behind another
/// build on a shared host.
fn run_metadata(workspace: &Path, env: &dyn Env) -> Result<CargoMetadata, BaselineError> {
    let cargo = env.string("CARGO").unwrap_or_else(|| "cargo".to_owned());
    let output = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--locked", "--no-deps"])
        .current_dir(workspace)
        .output()
        .map_err(|error| BaselineError::MetadataCommand {
            workspace: workspace.to_path_buf(),
            message: error.to_string(),
            remedy: "make the Cargo binary named by `CARGO` available on PATH",
        })?;
    if !output.status.success() {
        return Err(BaselineError::MetadataCommand {
            workspace: workspace.to_path_buf(),
            message: captured_stderr(&output.stderr),
            remedy: "run `cargo metadata --format-version 1 --locked` in the workspace and fix the reported error",
        });
    }

    parse_metadata(&output.stdout, workspace)
}

/// Run a tool version command and return its trimmed output.
fn run_tool_version(
    workspace: &Path,
    env: &dyn Env,
    tool: &'static str,
    arguments: &[&str],
) -> Result<String, BaselineError> {
    let executable = env
        .string(tool.to_ascii_uppercase().as_str())
        .unwrap_or_else(|| tool.to_owned());
    let output = Command::new(executable)
        .args(arguments)
        .current_dir(workspace)
        .output()
        .map_err(|error| BaselineError::ToolVersion {
            tool,
            message: error.to_string(),
            remedy: "make the tool available on PATH before recording a baseline",
        })?;
    if !output.status.success() {
        return Err(BaselineError::ToolVersion {
            tool,
            message: captured_stderr(&output.stderr),
            remedy: "make the tool available on PATH before recording a baseline",
        });
    }

    String::from_utf8(output.stdout)
        .map(|text| text.trim().to_owned())
        .map_err(|error| BaselineError::ToolVersion {
            tool,
            message: error.to_string(),
            remedy: "record the baseline on a host whose tool output is UTF-8",
        })
}

/// Return captured stderr, or a placeholder when the tool wrote nothing.
fn captured_stderr(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr).trim().to_owned();
    if text.is_empty() {
        "tool exited without a diagnostic".to_owned()
    } else {
        text
    }
}

#[cfg(test)]
mod tests;
