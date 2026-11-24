//! Walk a Criterion output tree and record the baselines it contains.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Deserialize;

use super::super::error::BaselineError;
use super::{BaselineCase, RequiredFiles};

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
pub(crate) fn discover_cases(
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
        reject_partial_baseline(&path, discovery)?;
        walk(&path, discovery)?;
    } else if file_type.is_file() {
        consider_file(&path, discovery)?;
    }

    Ok(())
}

/// Reject a baseline directory that holds measurements but no metadata.
///
/// Discovery keys cases off `benchmark.json`. A directory that holds only
/// `sample.json` or `estimates.json` would otherwise be skipped, and
/// compatibility would report the reference identifier as merely `removed`,
/// letting the lane pass without ever assessing the current measurement.
fn reject_partial_baseline(
    directory: &Path,
    discovery: &Discovery<'_>,
) -> Result<(), BaselineError> {
    if directory
        .file_name()
        .is_none_or(|name| name != discovery.baseline)
    {
        return Ok(());
    }
    if directory.join("benchmark.json").is_file() {
        return Ok(());
    }
    let has_measurement =
        directory.join("sample.json").is_file() || directory.join("estimates.json").is_file();
    if !has_measurement {
        return Ok(());
    }
    let identifier = directory.parent().and_then(Path::file_name).map_or_else(
        || "<unknown>".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );

    Err(BaselineError::MissingRequiredFile {
        id: identifier,
        path: directory.join("benchmark.json"),
        remedy: "delete the partial baseline and rerun the save step",
    })
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
    let estimates = recorded_baseline.join("estimates.json");
    let sample = recorded_baseline.join("sample.json");
    // A stale sample would be handed to Criterion as comparison input, so
    // every present file must also be newer than the marker, not just the
    // metadata that names the case.
    for present in [&estimates, &sample] {
        if present.is_file() {
            check_freshness(present, discovery.freshness)?;
        }
    }
    let required_files = RequiredFiles {
        benchmark: true,
        estimates: estimates.is_file(),
        sample: sample.is_file(),
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
