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
    let criterion_home = env.string(CRITERION_HOME_ENV).map(PathBuf::from);
    let cargo_target_dir = env.string(CARGO_TARGET_DIR_ENV).map(PathBuf::from);
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

/// Parsed subset of `cargo metadata` output.
#[derive(Debug, Deserialize)]
pub struct CargoMetadata {
    /// Directory Cargo writes build artefacts into.
    target_directory: PathBuf,
    /// Packages resolved for the workspace.
    #[serde(default)]
    packages: Vec<MetadataPackage>,
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

    /// Return the resolved version of the named package, if present.
    fn package_version(&self, name: &str) -> Option<&str> {
        self.packages
            .iter()
            .find(|package| package.name == name)
            .map(|package| package.version.as_str())
    }
}

/// One package entry from `cargo metadata`.
#[derive(Debug, Deserialize)]
struct MetadataPackage {
    /// Package name.
    name: String,
    /// Resolved package version.
    version: String,
}

/// Parse `cargo metadata` output.
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
    let metadata = run_metadata(workspace, env, false)?;
    Ok(metadata.target_directory)
}

/// Return the resolved Criterion version recorded by the workspace lockfile.
fn criterion_version(workspace: &Path, env: &dyn Env) -> Result<String, BaselineError> {
    let metadata = run_metadata(workspace, env, true)?;
    metadata
        .package_version("criterion")
        .map(str::to_owned)
        .ok_or_else(|| BaselineError::ToolVersion {
            tool: "criterion",
            message: format!(
                "no criterion package is resolved for {}",
                workspace.display()
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
/// # Examples
///
/// ```
/// # use mockable::DefaultEnv;
/// # use chutoro_test_support::ci::benchmark_baseline::output_root::record_tool_versions;
/// # let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
/// let tools = record_tool_versions(workspace, &DefaultEnv).expect("tool versions must run");
/// assert!(!tools.cargo.is_empty());
/// ```
pub fn record_tool_versions(
    workspace: &Path,
    env: &dyn Env,
) -> Result<RecordedTools, BaselineError> {
    Ok(RecordedTools {
        criterion: criterion_version(workspace, env)?,
        cargo: run_tool_version(workspace, env, "cargo", &["--version"])?,
        rustc: run_tool_version(workspace, env, "rustc", &["-Vv"])?,
    })
}

/// Run `cargo metadata` in the workspace and parse its output.
fn run_metadata(
    workspace: &Path,
    env: &dyn Env,
    include_dependencies: bool,
) -> Result<CargoMetadata, BaselineError> {
    let cargo = env.string("CARGO").unwrap_or_else(|| "cargo".to_owned());
    let mut command = Command::new(cargo);
    command
        .args(["metadata", "--format-version", "1", "--locked"])
        .current_dir(workspace);
    if !include_dependencies {
        command.arg("--no-deps");
    }

    let output = command
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
mod tests {
    //! Unit tests for output-root precedence and metadata parsing.

    use std::path::{Path, PathBuf};

    use super::{OutputRootInputs, parse_metadata, resolve_output_root};

    fn resolve(
        criterion_home: Option<&Path>,
        cargo_target_dir: Option<&Path>,
        metadata_target_dir: Option<&Path>,
    ) -> PathBuf {
        resolve_output_root(&OutputRootInputs {
            criterion_home,
            cargo_target_dir,
            metadata_target_dir,
            working_directory: Path::new("/work"),
        })
    }

    #[test]
    fn criterion_home_wins_over_every_other_source() {
        let root = resolve(
            Some(Path::new("/explicit")),
            Some(Path::new("/target")),
            Some(Path::new("/metadata")),
        );

        assert_eq!(root, Path::new("/explicit"));
    }

    #[test]
    fn relative_criterion_home_is_resolved_against_the_working_directory() {
        let root = resolve(Some(Path::new("custom/criterion")), None, None);

        assert_eq!(root, Path::new("/work/custom/criterion"));
    }

    #[test]
    fn cargo_target_directory_wins_over_metadata() {
        let root = resolve(
            None,
            Some(Path::new("/cargo-target")),
            Some(Path::new("/metadata")),
        );

        assert_eq!(root, Path::new("/cargo-target/criterion"));
    }

    #[test]
    fn relative_cargo_target_directory_is_resolved_against_the_working_directory() {
        let root = resolve(None, Some(Path::new("build")), None);

        assert_eq!(root, Path::new("/work/build/criterion"));
    }

    #[test]
    fn metadata_target_directory_is_used_when_no_override_exists() {
        let root = resolve(None, None, Some(Path::new("/metadata")));

        assert_eq!(root, Path::new("/metadata/criterion"));
    }

    #[test]
    fn default_root_matches_criterion_last_resort() {
        let root = resolve(None, None, None);

        assert_eq!(root, Path::new("/work/target/criterion"));
    }

    #[test]
    fn malformed_metadata_names_the_workspace() {
        let error = parse_metadata(b"{not json", Path::new("/reference"))
            .expect_err("malformed metadata must fail");

        let message = error.to_string();
        assert!(message.contains("/reference"), "{message}");
        assert!(message.contains("cargo metadata"), "{message}");
    }

    #[test]
    fn metadata_without_a_target_directory_is_rejected() {
        let error = parse_metadata(br#"{"packages":[]}"#, Path::new("/reference"))
            .expect_err("missing target directory must fail");

        assert!(error.to_string().contains("target_directory"), "{error}");
    }
}
