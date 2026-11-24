//! Record and read the baseline layout Criterion actually wrote.
//!
//! Criterion stores one directory per benchmark identifier beneath the output
//! root, then one directory per baseline name beneath that. The identifier
//! directory may contain separators (`hnsw_build/n=100,M=8,ef=16`), and
//! Criterion may append a collision suffix when two identifiers sanitize to the
//! same directory name. Mapping by the recorded `full_id` rather than by a
//! guessed path is what makes staging reliable.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

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
/// let environment = BTreeMap::new();
/// let request = InventoryRequest {
///     workspace: Path::new(env!("CARGO_MANIFEST_DIR")),
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

/// One identifier as recorded by Criterion in `benchmark.json`.
#[derive(Debug, Deserialize)]
struct CriterionBenchmarkId {
    /// Full benchmark identifier.
    full_id: String,
    /// Directory name Criterion assigned.
    directory_name: String,
}

/// Discovery state carried through the recursive walk.
struct Discovery<'a> {
    /// Output root being walked.
    root: &'a Path,
    /// Baseline directory name.
    baseline: &'a str,
    /// Optional freshness marker and its modification time.
    freshness: Option<(&'a Path, SystemTime)>,
    /// Cases found so far.
    cases: Vec<BaselineCase>,
    /// First path seen for each identifier.
    first_seen: HashMap<String, PathBuf>,
}

/// Discover every baseline entry for `baseline` beneath `root`.
fn discover_cases(
    root: &Path,
    baseline: &str,
    fresh_since: Option<&Path>,
) -> Result<Vec<BaselineCase>, BaselineError> {
    let freshness = fresh_since
        .map(|marker| marker_time(marker).map(|time| (marker, time)))
        .transpose()?;
    let mut discovery = Discovery {
        root,
        baseline,
        freshness,
        cases: Vec::new(),
        first_seen: HashMap::new(),
    };

    walk(root, &mut discovery)?;
    discovery
        .cases
        .sort_by(|left, right| left.full_id.cmp(&right.full_id));

    Ok(discovery.cases)
}

/// Return the modification time of the freshness marker.
fn marker_time(marker: &Path) -> Result<SystemTime, BaselineError> {
    fs::metadata(marker)
        .and_then(|metadata| metadata.modified())
        .map_err(|source| BaselineError::Read {
            path: marker.to_path_buf(),
            source,
            remedy: "write the freshness marker before measuring any baseline",
        })
}

/// Walk a directory recursively, recording baseline identifiers.
fn walk(directory: &Path, discovery: &mut Discovery<'_>) -> Result<(), BaselineError> {
    let entries = read_directory(directory)?;

    for entry in entries {
        let dir_entry = entry.map_err(|source| BaselineError::Read {
            path: directory.to_path_buf(),
            source,
            remedy: "check that the Criterion output root is readable",
        })?;
        visit(&dir_entry, discovery)?;
    }

    Ok(())
}

/// Read a directory, naming it when it cannot be read.
fn read_directory(directory: &Path) -> Result<fs::ReadDir, BaselineError> {
    fs::read_dir(directory).map_err(|source| BaselineError::Read {
        path: directory.to_path_buf(),
        source,
        remedy: "check that the Criterion output root exists and is readable",
    })
}

/// Visit one directory entry, recursing into directories.
fn visit(entry: &fs::DirEntry, discovery: &mut Discovery<'_>) -> Result<(), BaselineError> {
    let path = entry.path();
    let file_type = entry.file_type().map_err(|source| BaselineError::Read {
        path: path.clone(),
        source,
        remedy: "check that the Criterion output root is readable",
    })?;
    if file_type.is_dir() {
        walk(&path, discovery)?;
    } else if file_type.is_file() {
        consider_file(&path, discovery)?;
    }

    Ok(())
}

/// Return the identifier directory when `path` is a baseline `benchmark.json`.
fn identifier_directory<'a>(path: &'a Path, baseline: &str) -> Option<&'a Path> {
    if path.file_name().is_none_or(|name| name != "benchmark.json") {
        return None;
    }
    let baseline_directory = path.parent()?;
    if baseline_directory
        .file_name()
        .is_none_or(|name| name != baseline)
    {
        return None;
    }

    baseline_directory.parent()
}

/// Record a `benchmark.json` when it sits directly inside the baseline.
fn consider_file(path: &Path, discovery: &mut Discovery<'_>) -> Result<(), BaselineError> {
    let Some(identifier_directory) = identifier_directory(path, discovery.baseline) else {
        return Ok(());
    };

    check_freshness(path, discovery.freshness)?;
    let relative_directory = identifier_directory
        .strip_prefix(discovery.root)
        .map_err(|_| BaselineError::OutputRoot {
            reason: format!(
                "{} is outside the output root {}",
                identifier_directory.display(),
                discovery.root.display()
            ),
            remedy: "inventory the resolved Criterion output root, not its parent",
        })?;
    let bytes = fs::read(path).map_err(|source| BaselineError::Read {
        path: path.to_path_buf(),
        source,
        remedy: "delete the malformed baseline and rerun the save step",
    })?;
    let record: CriterionBenchmarkId =
        serde_json::from_slice(&bytes).map_err(|source| BaselineError::Parse {
            path: path.to_path_buf(),
            source,
            remedy: "delete the malformed baseline and rerun the save step",
        })?;

    if Path::new(&record.directory_name) != relative_directory {
        return Err(BaselineError::OutputRoot {
            reason: format!(
                "{} records directory {} but sits at {}",
                path.display(),
                record.directory_name,
                relative_directory.display()
            ),
            remedy: "rerun the save step; Criterion and the filesystem disagree",
        });
    }

    if let Some(first) = discovery.first_seen.get(&record.full_id) {
        return Err(BaselineError::DuplicateId {
            id: record.full_id,
            first: first.clone(),
            second: path.to_path_buf(),
            remedy: "delete the stale baseline tree and rerun the save step",
        });
    }
    discovery
        .first_seen
        .insert(record.full_id.clone(), path.to_path_buf());

    let recorded_baseline = path.parent().unwrap_or(path);
    let required_files = RequiredFiles {
        benchmark: true,
        estimates: recorded_baseline.join("estimates.json").is_file(),
        sample: recorded_baseline.join("sample.json").is_file(),
    };
    discovery.cases.push(BaselineCase {
        full_id: record.full_id,
        directory_name: record.directory_name,
        relative_directory: relative_directory.to_path_buf(),
        required_files,
    });

    Ok(())
}

/// Reject a baseline file older than the freshness marker.
fn check_freshness(
    path: &Path,
    freshness: Option<(&Path, SystemTime)>,
) -> Result<(), BaselineError> {
    let Some((marker, threshold)) = freshness else {
        return Ok(());
    };
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .map_err(|source| BaselineError::Read {
            path: path.to_path_buf(),
            source,
            remedy: "check that the baseline file is readable",
        })?;
    if modified < threshold {
        return Err(BaselineError::StaleBaseline {
            path: path.to_path_buf(),
            marker: marker.to_path_buf(),
            remedy: "delete the stale Criterion tree and rerun the save step",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests;
