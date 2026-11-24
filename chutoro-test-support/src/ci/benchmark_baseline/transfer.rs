//! Transfer reference baselines into the current Criterion tree.
//!
//! Criterion loads a baseline from the directory named by `--load-baseline`
//! and compares against the directory named by `--baseline`, both beneath the
//! current output root. Reference measurements therefore have to be copied
//! into the current tree before the comparison runs. Each identifier is copied
//! through a temporary sibling directory and renamed into place only after its
//! required files validate, so an interrupted copy cannot destroy the previous
//! destination.

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::{Builder, TempDir};

use super::error::BaselineError;
use super::preflight::ComparableCase;

/// Files Criterion needs to compare one identifier.
const REQUIRED_FILES: [&str; 3] = ["benchmark.json", "estimates.json", "sample.json"];

/// Inputs to the staging transfer.
#[derive(Clone, Copy, Debug)]
pub struct StageRequest<'a> {
    /// Resolved reference output root.
    pub reference_root: &'a Path,
    /// Resolved current output root.
    pub current_root: &'a Path,
    /// Reference baseline directory name.
    pub reference_baseline: &'a str,
    /// Identifiers present on both sides.
    pub comparable: &'a [ComparableCase],
}

/// One identifier whose reference data was staged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedCase {
    /// Criterion's full benchmark identifier.
    pub full_id: String,
    /// Destination baseline directory.
    pub destination: PathBuf,
}

/// Result of a successful staging transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageOutcome {
    /// Identifiers whose reference data is now in the current tree.
    pub staged: Vec<StagedCase>,
    /// Anchored, escaped filter selecting exactly those identifiers.
    pub filter: String,
}

/// Copy every comparable reference baseline into the current tree.
///
/// # Errors
///
/// Returns an error when a required source or copied file is missing, a copy
/// or rename fails, or stale change estimates cannot be removed. Existing
/// destinations are preserved when a copy fails.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use chutoro_test_support::ci::benchmark_baseline::preflight::ComparableCase;
/// use chutoro_test_support::ci::benchmark_baseline::transfer::{
///     StageRequest, stage,
/// };
///
/// let comparable = [ComparableCase {
///     full_id: "fixture/alpha".to_owned(),
///     reference_directory: "fixture/alpha".into(),
///     current_directory: "fixture/alpha".into(),
/// }];
/// let outcome = stage(&StageRequest {
///     reference_root: Path::new("/reference/target/criterion"),
///     current_root: Path::new("/current/target/criterion"),
///     reference_baseline: "ci-reference",
///     comparable: &comparable,
/// })
/// .expect("reference data must stage");
/// assert!(!outcome.filter.is_empty());
/// ```
pub fn stage(request: &StageRequest<'_>) -> Result<StageOutcome, BaselineError> {
    stage_with_fault(request, StageFault::None)
}

/// A copy fault used by unit tests to prove destination preservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StageFault {
    /// No fault is injected.
    None,
    /// Fail after the first file of the first identifier is copied.
    AfterFirstCopy,
}

/// Shared staging implementation with a test-only fault injection point.
fn stage_with_fault(
    request: &StageRequest<'_>,
    fault: StageFault,
) -> Result<StageOutcome, BaselineError> {
    let mut staged = Vec::new();
    for case in request.comparable {
        let source = request
            .reference_root
            .join(&case.reference_directory)
            .join(request.reference_baseline);
        let destination = request
            .current_root
            .join(&case.current_directory)
            .join(request.reference_baseline);
        let staged_case = stage_one(&source, &destination, &case.full_id, fault)?;
        staged.push(staged_case);
    }
    remove_stale_changes(request)?;

    let ids: Vec<String> = request
        .comparable
        .iter()
        .map(|case| case.full_id.clone())
        .collect();

    Ok(StageOutcome {
        staged,
        filter: build_filter(&ids),
    })
}

/// One source and destination pair being staged.
#[derive(Clone, Copy, Debug)]
struct StagePair<'a> {
    /// Reference baseline directory.
    source: &'a Path,
    /// Destination baseline directory.
    destination: &'a Path,
    /// Criterion's full benchmark identifier.
    id: &'a str,
}

/// Stage one identifier through a temporary sibling directory.
fn stage_one(
    source: &Path,
    destination: &Path,
    id: &str,
    fault: StageFault,
) -> Result<StagedCase, BaselineError> {
    let pair = StagePair {
        source,
        destination,
        id,
    };
    validate_files(pair.source, pair.id)?;
    let parent = destination
        .parent()
        .ok_or_else(|| BaselineError::StageCopy {
            id: id.to_owned(),
            source: source.to_path_buf(),
            destination: destination.to_path_buf(),
            message: "destination has no parent directory".to_owned(),
            remedy: "stage into a benchmark identifier directory beneath the output root",
        })?;
    fs::create_dir_all(parent).map_err(|source_error| BaselineError::StageCopy {
        id: id.to_owned(),
        source: source.to_path_buf(),
        destination: destination.to_path_buf(),
        message: source_error.to_string(),
        remedy: "ensure the current Criterion output root is writable",
    })?;

    let temporary = temporary_directory(parent, id)?;
    copy_files(&pair, temporary.path(), fault)?;
    validate_files(temporary.path(), id)?;
    replace_destination(parent, temporary, &pair)?;

    Ok(StagedCase {
        full_id: id.to_owned(),
        destination: destination.to_path_buf(),
    })
}

/// Create a temporary sibling directory beneath `parent`.
fn temporary_directory(parent: &Path, id: &str) -> Result<TempDir, BaselineError> {
    Builder::new()
        .prefix(".benchmark-baseline-stage-")
        .tempdir_in(parent)
        .map_err(|error| BaselineError::StageCopy {
            id: id.to_owned(),
            source: parent.to_path_buf(),
            destination: parent.to_path_buf(),
            message: error.to_string(),
            remedy: "ensure the current Criterion output root is writable",
        })
}

/// Copy each required file into the temporary directory.
fn copy_files(
    pair: &StagePair<'_>,
    destination: &Path,
    fault: StageFault,
) -> Result<(), BaselineError> {
    for (index, file) in REQUIRED_FILES.iter().enumerate() {
        let from = pair.source.join(file);
        let to = destination.join(file);
        fs::copy(&from, &to).map_err(|error| BaselineError::StageCopy {
            id: pair.id.to_owned(),
            source: from.clone(),
            destination: to.clone(),
            message: error.to_string(),
            remedy: "check that the reference baseline files are readable",
        })?;
        if fault == StageFault::AfterFirstCopy && index == 0 {
            return Err(BaselineError::StageCopy {
                id: pair.id.to_owned(),
                source: pair.source.to_path_buf(),
                destination: pair.destination.to_path_buf(),
                message: "injected copy failure".to_owned(),
                remedy: "this fault exists only to prove destination preservation",
            });
        }
    }

    Ok(())
}

/// Validate that every required file exists and carries content.
fn validate_files(directory: &Path, id: &str) -> Result<(), BaselineError> {
    for file in REQUIRED_FILES {
        let path = directory.join(file);
        let valid =
            fs::metadata(&path).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0);
        if !valid {
            return Err(BaselineError::MissingRequiredFile {
                id: id.to_owned(),
                path,
                remedy: "delete the partial baseline and rerun the save step",
            });
        }
    }

    Ok(())
}

/// Rename the temporary directory into place, preserving any previous one.
fn replace_destination(
    parent: &Path,
    temporary: TempDir,
    pair: &StagePair<'_>,
) -> Result<(), BaselineError> {
    let temporary_path = temporary.keep();
    let backup = if pair.destination.exists() {
        Some(move_to_backup(parent, pair)?)
    } else {
        None
    };

    if let Err(error) = fs::rename(&temporary_path, pair.destination) {
        let _cleanup = fs::remove_dir_all(&temporary_path);
        restore_backup(backup, pair.destination);
        return Err(BaselineError::StageCopy {
            id: pair.id.to_owned(),
            source: pair.source.to_path_buf(),
            destination: pair.destination.to_path_buf(),
            message: error.to_string(),
            remedy: "ensure the current Criterion output root is writable",
        });
    }

    Ok(())
}

/// Move an existing destination into a temporary backup directory.
fn move_to_backup(parent: &Path, pair: &StagePair<'_>) -> Result<TempDir, BaselineError> {
    let backup = Builder::new()
        .prefix(".benchmark-baseline-backup-")
        .tempdir_in(parent)
        .map_err(|error| BaselineError::StageCopy {
            id: pair.id.to_owned(),
            source: pair.source.to_path_buf(),
            destination: pair.destination.to_path_buf(),
            message: error.to_string(),
            remedy: "ensure the current Criterion output root is writable",
        })?;
    fs::rename(pair.destination, backup.path().join("previous")).map_err(|error| {
        BaselineError::StageCopy {
            id: pair.id.to_owned(),
            source: pair.source.to_path_buf(),
            destination: pair.destination.to_path_buf(),
            message: error.to_string(),
            remedy: "remove the stale destination baseline and rerun staging",
        }
    })?;

    Ok(backup)
}

/// Restore a backed-up destination after a failed rename.
fn restore_backup(backup: Option<TempDir>, destination: &Path) {
    if let Some(backup) = backup {
        let previous = backup.path().join("previous");
        let _restore_result = fs::rename(previous, destination);
    }
}

/// Remove stale comparison output for identifiers about to be compared.
fn remove_stale_changes(request: &StageRequest<'_>) -> Result<(), BaselineError> {
    for case in request.comparable {
        let change_directory = request
            .current_root
            .join(&case.current_directory)
            .join("change");
        if change_directory.exists() {
            fs::remove_dir_all(&change_directory).map_err(|error| BaselineError::Remove {
                path: change_directory.clone(),
                source: error,
                remedy: "ensure the current Criterion output root is writable",
            })?;
        }
    }

    Ok(())
}

/// Build an anchored, escaped regular expression selecting exactly `ids`.
///
/// Criterion matches its positional filter as a regular expression. Anchoring
/// and escaping is what keeps a current-only identifier from being selected,
/// which would make strict baseline comparison panic on a missing baseline.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::transfer::build_filter;
///
/// let filter = build_filter(&["fixture/alpha/n=1".to_owned(), "fixture/beta".to_owned()]);
/// assert_eq!(filter, "^(fixture/alpha/n=1|fixture/beta)$");
/// ```
#[must_use]
pub fn build_filter(ids: &[String]) -> String {
    let escaped: Vec<String> = ids.iter().map(|id| escape_regex(id)).collect();
    format!("^({})$", escaped.join("|"))
}

/// Escape the characters Criterion's regular expression treats specially.
fn escape_regex(id: &str) -> String {
    const SPECIAL: &str = r"\.^$*+?()[]{}|";
    let mut escaped = String::with_capacity(id.len());
    for character in id.chars() {
        if SPECIAL.contains(character) {
            escaped.push('\\');
        }
        escaped.push(character);
    }

    escaped
}

#[cfg(test)]
mod tests;
