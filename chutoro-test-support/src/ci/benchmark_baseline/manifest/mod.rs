//! Record and read the baseline layout Criterion actually wrote.
//!
//! Criterion stores one directory per benchmark identifier beneath the output
//! root, then one directory per baseline name beneath that. The identifier
//! directory may contain separators (`hnsw_build/n=100,M=8,ef=16`), and
//! Criterion may append a collision suffix when two identifiers sanitize to the
//! same directory name. Mapping by the recorded `full_id` rather than by a
//! guessed path is what makes staging reliable.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use mockable::Env;
use serde::{Deserialize, Serialize};

use super::error::BaselineError;
use super::output_root::record_tool_versions;
use super::{BENCH_ENV_PREFIX, SCHEMA_VERSION};

/// Versioned record of one Criterion baseline tree.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BaselineManifest {
    /// Schema version of this record.
    pub schema_version: u32,
    /// Revision the baseline was measured from.
    pub revision: String,
    /// Benchmark target that produced the baseline.
    pub bench: String,
    /// Criterion baseline name.
    pub baseline: String,
    /// Absolute Criterion output root.
    pub output_root: PathBuf,
    /// Tool versions that produced the baseline.
    pub tool_versions: ToolVersions,
    /// Measurement arguments, excluding baseline selection.
    pub bench_arguments: Vec<String>,
    /// Relevant `CHUTORO_BENCH_*` values at measurement time.
    pub environment: BTreeMap<String, String>,
    /// One entry per discovered benchmark identifier.
    pub cases: Vec<BaselineCase>,
}

/// Tool versions recorded with a baseline.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolVersions {
    /// Resolved Criterion version.
    pub criterion: String,
    /// `cargo --version` output.
    pub cargo: String,
    /// `rustc -Vv` output.
    pub rustc: String,
}

/// Presence of the files Criterion needs to compare one identifier.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RequiredFiles {
    /// `benchmark.json` identifies the benchmark.
    pub benchmark: bool,
    /// `estimates.json` carries the absolute measurements.
    pub estimates: bool,
    /// `sample.json` carries the raw sample used for comparison.
    pub sample: bool,
}

impl RequiredFiles {
    /// Return whether every required file is present.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::RequiredFiles;
    ///
    /// let complete = RequiredFiles {
    ///     benchmark: true,
    ///     estimates: true,
    ///     sample: true,
    /// };
    /// assert!(complete.is_complete());
    /// ```
    #[must_use]
    pub const fn is_complete(self) -> bool {
        self.benchmark && self.estimates && self.sample
    }
}

/// One benchmark identifier discovered in a baseline tree.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BaselineCase {
    /// Criterion's full benchmark identifier.
    pub full_id: String,
    /// Directory name Criterion assigned, including any collision suffix.
    pub directory_name: String,
    /// Identifier directory relative to the output root.
    pub relative_directory: PathBuf,
    /// Presence of the files Criterion needs to compare this identifier.
    pub required_files: RequiredFiles,
}

/// Inputs to baseline discovery.
#[derive(Clone, Copy, Debug)]
pub struct InventoryRequest<'a> {
    /// Workspace whose tool versions are recorded.
    pub workspace: &'a Path,
    /// Benchmark target name.
    pub bench: &'a str,
    /// Criterion baseline name.
    pub baseline: &'a str,
    /// Revision the baseline was measured from.
    pub revision: &'a str,
    /// Resolved Criterion output root.
    pub output_root: &'a Path,
    /// Marker file newer than which every baseline file must be.
    pub fresh_since: Option<&'a Path>,
    /// Measurement arguments, excluding baseline selection.
    pub bench_arguments: &'a [String],
    /// Relevant `CHUTORO_BENCH_*` values at measurement time.
    pub environment: &'a BTreeMap<String, String>,
}

/// Discover every baseline entry for `request.baseline` beneath the root.
///
/// # Errors
///
/// Returns an error when tool versions cannot be recorded, the output root cannot be walked, a baseline file is malformed or stale, or an identifier appears twice.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use std::path::Path;
/// use chutoro_test_support::ci::benchmark_baseline::manifest::{
///     InventoryRequest, inventory,
/// };
/// use mockable::DefaultEnv;
///
/// let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
/// let workspace = crate_directory.parent().expect("crate is in the workspace");
/// let environment = BTreeMap::new();
/// let request = InventoryRequest {
///     workspace,
///     bench: "example",
///     baseline: "ci-reference",
///     revision: "unknown",
///     output_root: Path::new(env!("CARGO_MANIFEST_DIR")),
///     fresh_since: None,
///     bench_arguments: &[],
///     environment: &environment,
/// };
/// // The repository root is not a Criterion tree, so discovery finds no case.
/// let manifest = inventory(&request, &DefaultEnv).expect("inventory must run");
/// assert!(manifest.cases.is_empty());
/// ```
pub fn inventory(
    request: &InventoryRequest<'_>,
    env: &dyn Env,
) -> Result<BaselineManifest, BaselineError> {
    let tools = record_tool_versions(request.workspace, env)?;
    let cases = discover_cases(request.output_root, request.baseline, request.fresh_since)?;

    Ok(BaselineManifest {
        schema_version: SCHEMA_VERSION,
        revision: request.revision.to_owned(),
        bench: request.bench.to_owned(),
        baseline: request.baseline.to_owned(),
        output_root: request.output_root.to_path_buf(),
        tool_versions: ToolVersions {
            criterion: tools.criterion,
            cargo: tools.cargo,
            rustc: tools.rustc,
        },
        bench_arguments: request.bench_arguments.to_vec(),
        environment: request.environment.clone(),
        cases,
    })
}

/// Collect the `CHUTORO_BENCH_*` environment values that affect a run.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::manifest::collect_bench_environment;
/// use mockable::DefaultEnv;
///
/// let environment = collect_bench_environment(&DefaultEnv);
/// assert!(environment.keys().all(|key| key.starts_with("CHUTORO_BENCH_")));
/// ```
#[must_use]
pub fn collect_bench_environment(env: &dyn Env) -> BTreeMap<String, String> {
    env.all()
        .into_iter()
        .filter(|(key, _)| key.starts_with(BENCH_ENV_PREFIX))
        .collect()
}

/// Write a manifest as pretty JSON, creating its parent directory.
///
/// # Errors
///
/// Returns an error when the manifest cannot be serialized or written.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use std::path::Path;
/// # use chutoro_test_support::ci::benchmark_baseline::manifest::{
/// #     BaselineManifest, ToolVersions, write_manifest,
/// # };
/// # let directory = tempfile::tempdir().expect("temporary directory");
/// let manifest = BaselineManifest {
///     schema_version: 1,
///     revision: "abc".to_owned(),
///     bench: "hnsw".to_owned(),
///     baseline: "ci-reference".to_owned(),
///     output_root: Path::new("/work/target/criterion").to_path_buf(),
///     tool_versions: ToolVersions {
///         criterion: "0.5.1".to_owned(),
///         cargo: "cargo 1.99.1".to_owned(),
///         rustc: "rustc 1.99.1".to_owned(),
///     },
///     bench_arguments: Vec::new(),
///     environment: BTreeMap::new(),
///     cases: Vec::new(),
/// };
/// let path = directory.path().join("manifest.json");
/// write_manifest(&path, &manifest).expect("manifest must be written");
/// assert!(path.is_file());
/// ```
pub fn write_manifest(path: &Path, manifest: &BaselineManifest) -> Result<(), BaselineError> {
    write_json(path, manifest)
}

/// Read a manifest, naming the path when it cannot be parsed.
///
/// # Errors
///
/// Returns an error when the manifest cannot be read or parsed.
pub fn read_manifest(path: &Path) -> Result<BaselineManifest, BaselineError> {
    read_json(path)
}

/// Write any serializable value as pretty JSON.
pub(crate) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), BaselineError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| BaselineError::Serialize {
        path: path.to_path_buf(),
        source,
        remedy: "report the serialization failure; the value should be representable as JSON",
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| BaselineError::Write {
            path: parent.to_path_buf(),
            source,
            remedy: "ensure the evidence directory is writable",
        })?;
    }

    fs::write(path, bytes).map_err(|source| BaselineError::Write {
        path: path.to_path_buf(),
        source,
        remedy: "ensure the evidence directory is writable",
    })
}

/// Read any deserializable value from JSON.
pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, BaselineError> {
    let bytes = fs::read(path).map_err(|source| BaselineError::Read {
        path: path.to_path_buf(),
        source,
        remedy: "check that the previous lane step wrote the file",
    })?;
    serde_json::from_slice(&bytes).map_err(|source| BaselineError::Parse {
        path: path.to_path_buf(),
        source,
        remedy: "rerun the step that wrote the file; a partial write is a lane defect",
    })
}

mod discovery;

pub(crate) use discovery::discover_cases;

#[cfg(test)]
mod tests;
