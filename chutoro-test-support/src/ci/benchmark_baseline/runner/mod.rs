//! Orchestrate the helper's three stages behind a thin binary.
//!
//! The binary in `src/bin/benchmark_baseline.rs` only parses arguments, calls
//! one of these functions, and maps the result to an exit code and GitHub
//! output. Keeping the orchestration here is what lets the lane's behaviour be
//! tested without spawning a process, and what keeps the binary free of
//! policy.

use std::collections::BTreeMap;
use std::path::Path;

use mockable::DefaultEnv;

use super::args::{InventoryArgs, StageArgs};
use super::compatibility::{Compatibility, check};
use super::manifest::{
    BaselineManifest, collect_bench_environment, inventory, read_manifest, write_manifest,
};
use super::output_root::resolve_workspace_output_root;
use super::preflight::{ManifestSummary, PreflightReport};
use super::transfer::{StageOutcome, StageRequest, stage};
use super::{BaselineError, Outcome};

/// Name of the inventory manifest written into the evidence directory.
pub const REFERENCE_MANIFEST_FILE: &str = "reference-manifest.json";

/// Name of the current inventory manifest written into the evidence directory.
pub const CURRENT_MANIFEST_FILE: &str = "current-manifest.json";

/// Name of the preflight report written by `stage`.
pub const PREFLIGHT_FILE: &str = "preflight.json";

mod classify;

pub use classify::run_classify;

/// Run `inventory` and write the manifest.
///
/// # Errors
///
/// Returns an error when the output root cannot be resolved, tool versions
/// cannot be recorded, or the manifest cannot be written.
///
/// # Examples
///
/// ```no_run
/// use chutoro_test_support::ci::benchmark_baseline::args::InventoryArgs;
/// use chutoro_test_support::ci::benchmark_baseline::runner::run_inventory;
///
/// let manifest = run_inventory(&InventoryArgs {
///     workspace: ".".into(),
///     bench: "hnsw".to_owned(),
///     baseline: "ci-reference".to_owned(),
///     revision: "abc".to_owned(),
///     output: "/evidence/reference-manifest.json".into(),
///     fresh_since: None,
///     bench_arguments: vec!["--noplot".to_owned()],
/// })
/// .expect("inventory must succeed");
/// assert_eq!(manifest.bench, "hnsw");
/// ```
pub fn run_inventory(args: &InventoryArgs) -> Result<BaselineManifest, BaselineError> {
    let env = DefaultEnv;
    let output_root = resolve_workspace_output_root(&args.workspace, &env)?;
    let environment: BTreeMap<String, String> = collect_bench_environment(&env);
    let manifest = inventory(
        &super::manifest::InventoryRequest {
            workspace: &args.workspace,
            bench: &args.bench,
            baseline: &args.baseline,
            revision: &args.revision,
            output_root: &output_root,
            fresh_since: args.fresh_since.as_deref(),
            bench_arguments: &args.bench_arguments,
            environment: &environment,
        },
        &env,
    )?;
    write_manifest(&args.output, &manifest)?;

    Ok(manifest)
}

/// Run `stage`, writing the preflight report and the comparison filter.
///
/// # Errors
///
/// Returns an error when either manifest cannot be read, the two sides are not
/// comparable, or the transfer fails. A preflight report is written before the
/// error is returned, so the workflow always has the diagnosis.
///
/// # Examples
///
/// ```no_run
/// use chutoro_test_support::ci::benchmark_baseline::args::StageArgs;
/// use chutoro_test_support::ci::benchmark_baseline::runner::run_stage;
///
/// let report = run_stage(&StageArgs {
///     bench: "hnsw".to_owned(),
///     reference_manifest: "/evidence/reference-manifest.json".into(),
///     current_manifest: "/evidence/current-manifest.json".into(),
///     evidence: "/evidence".into(),
///     filter_output: Some("/evidence/filter.txt".into()),
/// })
/// .expect("staging must succeed");
/// assert!(!report.comparable.is_empty());
/// ```
pub fn run_stage(args: &StageArgs) -> Result<PreflightReport, BaselineError> {
    let report_path = args.evidence.join(PREFLIGHT_FILE);
    // The report is published before either manifest is read, so a missing or
    // malformed manifest still leaves a diagnosis rather than none.
    let mut report = PreflightReport::new("stage", &args.bench);
    report.write(&report_path)?;

    let reference = read_stage_manifest(
        &args.reference_manifest,
        "reference",
        &mut report,
        &report_path,
    )?;
    let current =
        read_stage_manifest(&args.current_manifest, "current", &mut report, &report_path)?;
    report.reference = Some(ManifestSummary::from_manifest(&reference));
    report.current = Some(ManifestSummary::from_manifest(&current));
    check_requested_bench(&reference, &current, &args.bench)
        .map_err(|error| record_stage_failure(&mut report, &report_path, "compatibility", error))?;

    let compatibility = match check(&reference, &current) {
        Ok(compatibility) => compatibility,
        Err(error) => {
            report.outcome = error.outcome().unwrap_or(Outcome::Incomplete);
            report.reason = Some(error.to_string());
            report
                .diagnostics
                .push(error.diagnostic().with_stage("compatibility"));
            report.write(&report_path)?;
            return Err(error);
        }
    };

    let outcome = match stage(&StageRequest {
        reference_root: &reference.output_root,
        current_root: &current.output_root,
        reference_baseline: &reference.baseline,
        comparable: &compatibility.comparable,
    }) {
        Ok(outcome) => outcome,
        Err(error) => {
            report.outcome = error.outcome().unwrap_or(Outcome::Incomplete);
            report.reason = Some(error.to_string());
            report
                .diagnostics
                .push(error.diagnostic().with_stage("stage"));
            report.write(&report_path)?;
            return Err(error);
        }
    };

    // Write the filter before publishing a passing report: classification
    // treats `Pass` as permission to use the comparison, so a filter that
    // could not be written must not leave that permission behind.
    if let Some(error) = args
        .filter_output
        .as_deref()
        .and_then(|path| write_filter(path, &outcome.filter).err())
    {
        return Err(record_stage_failure(
            &mut report,
            &report_path,
            "stage",
            error,
        ));
    }
    record_success(&mut report, &compatibility, &outcome);
    report.write(&report_path)?;

    Ok(report)
}

/// Reject manifests recorded for a different benchmark target.
fn check_requested_bench(
    reference: &BaselineManifest,
    current: &BaselineManifest,
    requested: &str,
) -> Result<(), BaselineError> {
    for manifest in [reference, current] {
        if manifest.bench != requested {
            return Err(BaselineError::Incompatible {
                field: "requested benchmark target",
                reference: manifest.bench.clone(),
                current: requested.to_owned(),
                remedy: "pass the --bench the manifests were recorded for",
            });
        }
    }

    Ok(())
}

/// Read one side's manifest, recording a failure in the preflight report.
fn read_stage_manifest(
    path: &Path,
    side: &str,
    report: &mut PreflightReport,
    report_path: &Path,
) -> Result<BaselineManifest, BaselineError> {
    read_manifest(path).map_err(|error| record_stage_failure(report, report_path, side, error))
}

/// Record a staging failure in the preflight report and return the error.
fn record_stage_failure(
    report: &mut PreflightReport,
    report_path: &Path,
    stage: &str,
    error: BaselineError,
) -> BaselineError {
    report.outcome = error.outcome().unwrap_or(Outcome::Incomplete);
    report.reason = Some(error.to_string());
    let diagnostic = error.diagnostic().with_stage(stage);
    report.diagnostics.push(diagnostic);
    // A failed evidence write must not mask the original failure, so the
    // write error is logged rather than returned.
    if let Err(write_error) = report.write(report_path) {
        tracing::warn!(
            path = %report_path.display(),
            error = %write_error,
            "could not write the preflight report for a failed stage",
        );
    }
    error
}

/// Record a successful transfer in the preflight report.
fn record_success(
    report: &mut PreflightReport,
    compatibility: &Compatibility,
    outcome: &StageOutcome,
) {
    report.outcome = Outcome::Pass;
    report.comparable.clone_from(&compatibility.comparable);
    report
        .new_unverified
        .clone_from(&compatibility.new_unverified);
    report.removed.clone_from(&compatibility.removed);
    if !report.new_unverified.is_empty() {
        report.diagnostics.push(super::BaselineDiagnostic {
            stage: String::from("compatibility"),
            outcome: Outcome::Pass,
            id: None,
            path: None,
            expected: None,
            remedy: String::from(
                "save a reference baseline for these identifiers before relying on them",
            ),
            message: format!(
                "{} current-only identifier(s) were not compared",
                report.new_unverified.len()
            ),
        });
    }
    for case in &outcome.staged {
        tracing::debug!(
            id = %case.full_id,
            destination = %case.destination.display(),
            "staged reference baseline",
        );
    }
}

/// Write the comparison filter, appending a newline for shell substitution.
fn write_filter(path: &Path, filter: &str) -> Result<(), BaselineError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| BaselineError::Write {
            path: parent.to_path_buf(),
            source,
            remedy: "ensure the evidence directory is writable",
        })?;
    }

    std::fs::write(path, format!("{filter}\n")).map_err(|source| BaselineError::Write {
        path: path.to_path_buf(),
        source,
        remedy: "ensure the evidence directory is writable",
    })
}

/// Resolve the output root for a workspace using the process environment.
///
/// # Errors
///
/// Returns an error when the root cannot be resolved.
///
/// # Examples
///
/// ```no_run
/// use chutoro_test_support::ci::benchmark_baseline::runner::workspace_output_root;
///
/// let root = workspace_output_root(std::path::Path::new(".")).expect("root resolves");
/// assert!(root.ends_with("criterion"));
/// ```
pub fn workspace_output_root(workspace: &Path) -> Result<std::path::PathBuf, BaselineError> {
    resolve_workspace_output_root(workspace, &DefaultEnv)
}

#[cfg(test)]
mod tests;
