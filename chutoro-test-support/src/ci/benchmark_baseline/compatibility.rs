//! Decide whether two baseline trees can be compared.
//!
//! Compatibility is a pure check over two manifests. It rejects different
//! schemas, Criterion versions, Rust toolchains, benchmark targets,
//! measurement arguments, and benchmark environment values, because a
//! comparison across any of those measures the configuration rather than the
//! code. Identifiers present on only one side are reported but do not fail:
//! they cannot be compared, and hiding them would make a shrinking benchmark
//! set look healthy.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use super::manifest::{BaselineCase, BaselineManifest};
use super::preflight::ComparableCase;
use super::{BaselineError, SUPPORTED_CRITERION_VERSION};

/// Result of a successful compatibility check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Compatibility {
    /// Identifiers present on both sides, sorted by identifier.
    pub comparable: Vec<ComparableCase>,
    /// Identifiers present only on the current side, sorted.
    pub new_unverified: Vec<String>,
    /// Identifiers present only on the reference side, sorted.
    pub removed: Vec<String>,
}

/// Check that a reference manifest and a current manifest can be compared.
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::benchmark_baseline::compatibility::check;
/// use chutoro_test_support::ci::benchmark_baseline::{
///     BaselineCase, BaselineManifest, RequiredFiles, ToolVersions,
/// };
/// use std::collections::BTreeMap;
/// use std::path::Path;
///
/// fn manifest(baseline: &str, full_id: &str) -> BaselineManifest {
///     BaselineManifest {
///         schema_version: 1,
///         revision: "abc".to_owned(),
///         bench: "hnsw".to_owned(),
///         baseline: baseline.to_owned(),
///         output_root: Path::new("/work/target/criterion").to_path_buf(),
///         tool_versions: ToolVersions {
///             criterion: "0.5.1".to_owned(),
///             cargo: "cargo 1.99.1".to_owned(),
///             rustc: "rustc 1.99.1".to_owned(),
///         },
///         bench_arguments: vec!["--noplot".to_owned()],
///         environment: BTreeMap::new(),
///         cases: vec![BaselineCase {
///             full_id: full_id.to_owned(),
///             directory_name: full_id.to_owned(),
///             relative_directory: full_id.into(),
///             required_files: RequiredFiles {
///                 benchmark: true,
///                 estimates: true,
///                 sample: true,
///             },
///         }],
///     }
/// }
///
/// let reference = manifest("ci-reference", "fixture/alpha");
/// let current = manifest("ci-current", "fixture/alpha");
/// let compatibility = check(&reference, &current).expect("manifests are compatible");
/// assert_eq!(compatibility.comparable.len(), 1);
/// ```
pub fn check(
    reference: &BaselineManifest,
    current: &BaselineManifest,
) -> Result<Compatibility, BaselineError> {
    check_schemas(reference, current)?;
    check_tool_versions(reference, current)?;
    check_target(reference, current)?;
    check_baseline_names(reference, current)?;

    let mut reference_cases = case_map(reference)?;
    let mut current_cases = case_map(current)?;
    let mut comparable = Vec::new();
    let mut new_unverified = Vec::new();
    let mut removed = Vec::new();

    for (full_id, current_case) in &current_cases {
        match reference_cases.remove(full_id) {
            Some(reference_case) => comparable.push(ComparableCase {
                full_id: full_id.clone(),
                reference_directory: reference_case.relative_directory.clone(),
                current_directory: current_case.relative_directory.clone(),
            }),
            None => new_unverified.push(full_id.clone()),
        }
    }
    removed.extend(reference_cases.into_keys());
    comparable.sort_by(|left, right| left.full_id.cmp(&right.full_id));
    new_unverified.sort();
    removed.sort();

    if comparable.is_empty() {
        return Err(BaselineError::EmptyComparable {
            bench: reference.bench.clone(),
            reference_baseline: reference.baseline.clone(),
            current_baseline: current.baseline.clone(),
            remedy: "confirm that both measurements ran the same benchmark target and filter",
        });
    }

    check_required_files(reference, current, &comparable)?;

    Ok(Compatibility {
        comparable,
        new_unverified,
        removed,
    })
}

/// Check that both manifests use the schema this helper understands.
fn check_schemas(
    reference: &BaselineManifest,
    current: &BaselineManifest,
) -> Result<(), BaselineError> {
    for manifest in [reference, current] {
        if manifest.schema_version != super::SCHEMA_VERSION {
            return Err(BaselineError::Schema {
                path: manifest.output_root.join("manifest.json"),
                found: manifest.schema_version,
                expected: super::SCHEMA_VERSION,
                remedy: "regenerate both manifests with this helper version",
            });
        }
    }

    Ok(())
}

/// Check Criterion, Rust, target, arguments, and environment compatibility.
fn check_tool_versions(
    reference: &BaselineManifest,
    current: &BaselineManifest,
) -> Result<(), BaselineError> {
    if reference.tool_versions.criterion != SUPPORTED_CRITERION_VERSION
        || current.tool_versions.criterion != SUPPORTED_CRITERION_VERSION
    {
        return Err(BaselineError::Incompatible {
            field: "criterion version support",
            reference: reference.tool_versions.criterion.clone(),
            current: current.tool_versions.criterion.clone(),
            remedy: "validate the pinned Criterion layout before changing this helper",
        });
    }
    require_equal(
        "criterion version",
        &reference.tool_versions.criterion,
        &current.tool_versions.criterion,
    )?;
    require_equal(
        "rustc -Vv",
        &reference.tool_versions.rustc,
        &current.tool_versions.rustc,
    )?;

    Ok(())
}

/// Check the benchmark target, arguments, environment, and baseline names.
fn check_target(
    reference: &BaselineManifest,
    current: &BaselineManifest,
) -> Result<(), BaselineError> {
    require_equal("benchmark target", &reference.bench, &current.bench)?;
    require_equal(
        "benchmark arguments",
        &reference.bench_arguments.join(" "),
        &current.bench_arguments.join(" "),
    )?;
    let (reference_environment, current_environment) =
        differing_environment(&reference.environment, &current.environment);
    if !reference_environment.is_empty() || !current_environment.is_empty() {
        return Err(BaselineError::Incompatible {
            field: "CHUTORO_BENCH_* environment",
            reference: reference_environment,
            current: current_environment,
            remedy: "set the same benchmark environment values for both measurements",
        });
    }

    Ok(())
}

/// Reject staged data that would overwrite the current measurement.
fn check_baseline_names(
    reference: &BaselineManifest,
    current: &BaselineManifest,
) -> Result<(), BaselineError> {
    if reference.baseline == current.baseline {
        return Err(BaselineError::Incompatible {
            field: "baseline names",
            reference: reference.baseline.clone(),
            current: current.baseline.clone(),
            remedy: "save the reference and current measurements under distinct baseline names",
        });
    }

    Ok(())
}

/// Return an error when two values differ.
fn require_equal(field: &'static str, reference: &str, current: &str) -> Result<(), BaselineError> {
    if reference == current {
        return Ok(());
    }

    Err(BaselineError::Incompatible {
        field,
        reference: reference.to_owned(),
        current: current.to_owned(),
        remedy: "measure both revisions with the same toolchain, target, and arguments",
    })
}

/// Render only the environment keys whose values differ.
fn differing_environment(
    reference: &BTreeMap<String, String>,
    current: &BTreeMap<String, String>,
) -> (String, String) {
    let mut keys: Vec<&String> = reference.keys().chain(current.keys()).collect();
    keys.sort();
    keys.dedup();

    let mut reference_parts = Vec::new();
    let mut current_parts = Vec::new();
    for key in keys {
        let reference_value = reference.get(key);
        let current_value = current.get(key);
        if reference_value == current_value {
            continue;
        }
        reference_parts.push(format!(
            "{key}={}",
            reference_value.map_or("<unset>", String::as_str)
        ));
        current_parts.push(format!(
            "{key}={}",
            current_value.map_or("<unset>", String::as_str)
        ));
    }

    (reference_parts.join("; "), current_parts.join("; "))
}

/// Build a map from identifier to case, rejecting duplicates.
fn case_map<'a>(
    manifest: &'a BaselineManifest,
) -> Result<HashMap<String, &'a BaselineCase>, BaselineError> {
    let mut cases = HashMap::new();
    for case in &manifest.cases {
        match cases.entry(case.full_id.clone()) {
            Entry::Occupied(existing) => {
                return Err(BaselineError::DuplicateId {
                    id: case.full_id.clone(),
                    first: case_path(manifest, existing.get()),
                    second: case_path(manifest, case),
                    remedy: "delete the stale baseline tree and rerun the save step",
                });
            }
            Entry::Vacant(slot) => {
                slot.insert(case);
            }
        }
    }

    Ok(cases)
}

/// Return the `benchmark.json` path for one case.
fn case_path(manifest: &BaselineManifest, case: &BaselineCase) -> PathBuf {
    manifest
        .output_root
        .join(&case.relative_directory)
        .join(&manifest.baseline)
        .join("benchmark.json")
}

/// Reject comparable identifiers that lack files Criterion needs.
fn check_required_files(
    reference: &BaselineManifest,
    current: &BaselineManifest,
    comparable: &[ComparableCase],
) -> Result<(), BaselineError> {
    let reference_cases = case_lookup(reference);
    let current_cases = case_lookup(current);
    for case in comparable {
        check_one_side(
            reference,
            reference_cases.get(case.full_id.as_str()).copied(),
            &case.full_id,
        )?;
        check_one_side(
            current,
            current_cases.get(case.full_id.as_str()).copied(),
            &case.full_id,
        )?;
    }

    Ok(())
}

/// Index a manifest's cases by identifier.
fn case_lookup(manifest: &BaselineManifest) -> HashMap<&str, &BaselineCase> {
    manifest
        .cases
        .iter()
        .map(|case| (case.full_id.as_str(), case))
        .collect()
}

/// Reject one side of a comparable identifier when a file is absent.
fn check_one_side(
    manifest: &BaselineManifest,
    case: Option<&BaselineCase>,
    full_id: &str,
) -> Result<(), BaselineError> {
    let Some(case) = case else {
        return Err(BaselineError::MissingRequiredFile {
            id: full_id.to_owned(),
            path: manifest.output_root.join(&manifest.baseline),
            remedy: "rerun inventory and staging against the same manifests",
        });
    };
    let required = case.required_files;
    let baseline_directory = manifest
        .output_root
        .join(&case.relative_directory)
        .join(&manifest.baseline);
    for (present, file) in [
        (required.benchmark, "benchmark.json"),
        (required.estimates, "estimates.json"),
        (required.sample, "sample.json"),
    ] {
        if !present {
            return Err(BaselineError::MissingRequiredFile {
                id: case.full_id.clone(),
                path: baseline_directory.join(file),
                remedy: "delete the partial baseline and rerun the save step",
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
