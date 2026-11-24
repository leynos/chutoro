//! Orchestrate the helper's three stages behind a thin binary.
//!
//! The binary in `src/bin/benchmark_baseline.rs` only parses arguments, calls
//! one of these functions, and maps the result to an exit code and GitHub
//! output. Keeping the orchestration here is what lets the lane's behaviour be
//! tested without spawning a process, and what keeps the binary free of
//! policy.

use std::collections::BTreeMap;
use std::path::Path;

use mockable::{DefaultEnv, Env};

use super::args::{ClassifyArgs, InventoryArgs, NoReferenceReason, StageArgs};
use super::classification::{Classification, read_case_change};
use super::compatibility::{Compatibility, check};
use super::manifest::{
    BaselineManifest, collect_bench_environment, inventory, read_manifest, write_manifest,
};
use super::output_root::resolve_workspace_output_root;
use super::preflight::{ComparableCase, ManifestSummary, PreflightReport};
use super::transfer::{StageOutcome, StageRequest, stage};
use super::verdict::{BaselineVerdict, CaseClass, CaseVerdict, NoComparison};
use super::{BaselineError, Outcome};

/// Name of the inventory manifest written into the evidence directory.
pub const REFERENCE_MANIFEST_FILE: &str = "reference-manifest.json";

/// Name of the current inventory manifest written into the evidence directory.
pub const CURRENT_MANIFEST_FILE: &str = "current-manifest.json";

/// Name of the preflight report written by `stage`.
pub const PREFLIGHT_FILE: &str = "preflight.json";

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
    let reference = read_manifest(&args.reference_manifest)?;
    let current = read_manifest(&args.current_manifest)?;
    let report_path = args.evidence.join(PREFLIGHT_FILE);

    let mut report = PreflightReport::new("stage", &args.bench);
    report.reference = Some(ManifestSummary::from_manifest(&reference));
    report.current = Some(ManifestSummary::from_manifest(&current));

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

    record_success(&mut report, &compatibility, &outcome);
    report.write(&report_path)?;
    if let Some(path) = &args.filter_output {
        write_filter(path, &outcome.filter)?;
    }

    Ok(report)
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

/// Run `classify` and write the verdict.
///
/// # Errors
///
/// Returns an error when the preflight report cannot be read or the verdict
/// cannot be written. A regression is reported in the verdict, not as an
/// error.
///
/// # Examples
///
/// ```no_run
/// use chutoro_test_support::ci::benchmark_baseline::args::ClassifyArgs;
/// use chutoro_test_support::ci::benchmark_baseline::runner::run_classify;
///
/// let verdict = run_classify(&ClassifyArgs {
///     preflight: "/evidence/preflight.json".into(),
///     bench: "hnsw".to_owned(),
///     verdict: "/evidence/verdict.json".into(),
///     threshold: 0.05,
///     no_reference: None,
/// })
/// .expect("classification must succeed");
/// assert!(verdict.comparable_count > 0);
/// ```
pub fn run_classify(args: &ClassifyArgs) -> Result<BaselineVerdict, BaselineError> {
    // A lane defect may have happened before staging wrote a preflight
    // report, so that path does not require one to exist.
    let mut verdict = match &args.failed_stage {
        Some(stage) => {
            let mut verdict = BaselineVerdict::started(&args.bench, args.threshold);
            verdict.stage = String::from("classify");
            record_lane_defect(&mut verdict, stage, args.failure_reason.as_deref());
            verdict
        }
        None => classify_by_state(args)?,
    };
    verdict.bench.clone_from(&args.bench);
    verdict.write(&args.verdict)?;
    write_step_summary(&verdict)?;

    Ok(verdict)
}

/// Choose the classification path the supplied arguments ask for.
fn classify_by_state(args: &ClassifyArgs) -> Result<BaselineVerdict, BaselineError> {
    args.no_reference.map_or_else(
        || classify_preflight(args),
        |reason| Ok(no_reference_verdict(args, reason)),
    )
}

/// Record a lane defect over whatever the preflight report said.
///
/// The workflow calls this from its failure handler so the evidence directory
/// always names the stage that failed, even when the failure happened before
/// any classification could run.
fn record_lane_defect(verdict: &mut BaselineVerdict, stage: &str, reason: Option<&str>) {
    verdict.outcome = Outcome::Incomplete;
    verdict.stage = String::from("classify");
    verdict.failed_stage = Some(stage.to_owned());
    let message = reason.map_or_else(
        || format!("stage {stage} failed before a comparison could run"),
        str::to_owned,
    );
    verdict.reason = Some(message.clone());
    verdict.diagnostics.push(super::BaselineDiagnostic {
        stage: stage.to_owned(),
        outcome: Outcome::Incomplete,
        id: None,
        path: None,
        expected: None,
        remedy: "read the stage log in the uploaded evidence directory".to_owned(),
        message,
    });
}

/// Append the verdict's Markdown table to `GITHUB_STEP_SUMMARY` when set.
fn write_step_summary(verdict: &BaselineVerdict) -> Result<(), BaselineError> {
    use std::io::Write as _;

    let env = DefaultEnv;
    let Some(path) = env.string("GITHUB_STEP_SUMMARY") else {
        return Ok(());
    };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|source| BaselineError::Write {
            path: path.clone().into(),
            source,
            remedy: "ensure GITHUB_STEP_SUMMARY names a writable file",
        })?;
    file.write_all(verdict.to_markdown().as_bytes())
        .map_err(|source| BaselineError::Write {
            path: path.into(),
            source,
            remedy: "ensure GITHUB_STEP_SUMMARY names a writable file",
        })
}

/// Build the verdict for a lane with no previous revision.
fn no_reference_verdict(args: &ClassifyArgs, reason: NoReferenceReason) -> BaselineVerdict {
    let remedy = match reason {
        NoReferenceReason::Shallow => {
            "check out with `fetch-depth: 2`, or fetch the previous commit before comparing"
        }
        NoReferenceReason::RootCommit => {
            "the first commit has no parent; the lane starts comparing from the next revision"
        }
    };
    let description = format!("no reference commit exists ({})", reason.as_str());
    let mut verdict = BaselineVerdict::without_comparison(&NoComparison {
        bench: &args.bench,
        outcome: Outcome::NoReference,
        stage: "classify",
        reason: &description,
        threshold: args.threshold,
    });
    if let Some(diagnostic) = verdict.diagnostics.first_mut() {
        diagnostic.remedy = String::from(remedy);
    }

    verdict
}

/// Classify every comparable identifier named by the preflight report.
fn classify_preflight(args: &ClassifyArgs) -> Result<BaselineVerdict, BaselineError> {
    let report = PreflightReport::read(&args.preflight)?;
    let mut verdict = BaselineVerdict::started(&args.bench, args.threshold);
    verdict.stage = String::from("classify");
    verdict.reference_revision = report
        .reference
        .as_ref()
        .map(|summary| summary.revision.clone());
    verdict.current_revision = report
        .current
        .as_ref()
        .map(|summary| summary.revision.clone());
    verdict.comparable_count = report.comparable.len();
    verdict.diagnostics.clone_from(&report.diagnostics);

    if report.outcome != Outcome::Pass {
        verdict.outcome = report.outcome;
        verdict.reason = report
            .reason
            .clone()
            .or_else(|| Some("staging did not produce comparable evidence".to_owned()));
        verdict.failed_stage = Some(report.stage.clone());
        record_unverified(&mut verdict, &report);
        verdict.cases.sort_by(|left, right| left.id.cmp(&right.id));
        return Ok(verdict);
    }

    let Some(current_root) = report
        .current
        .as_ref()
        .map(|summary| summary.output_root.clone())
    else {
        verdict.outcome = Outcome::Incomplete;
        verdict.failed_stage = Some("classify".to_owned());
        verdict.reason = Some("the preflight report recorded no current output root".to_owned());
        return Ok(verdict);
    };

    let mut unreadable = 0;
    for case in &report.comparable {
        match classify_case(&current_root, case, args.threshold) {
            Ok(case_verdict) => verdict.cases.push(case_verdict),
            Err(error) => {
                unreadable += 1;
                verdict
                    .diagnostics
                    .push(error.diagnostic().with_stage("classify"));
                verdict.cases.push(CaseVerdict::not_compared(
                    &case.full_id,
                    CaseClass::Unreadable,
                    &error.to_string(),
                ));
            }
        }
    }
    record_unverified(&mut verdict, &report);
    verdict.cases.sort_by(|left, right| left.id.cmp(&right.id));
    verdict.outcome = outcome_for(&verdict, unreadable);
    verdict.reason = reason_for(&verdict, unreadable);

    Ok(verdict)
}

/// Append identifiers the lane could not compare.
fn record_unverified(verdict: &mut BaselineVerdict, report: &PreflightReport) {
    for id in &report.new_unverified {
        verdict.cases.push(CaseVerdict::not_compared(
            id,
            CaseClass::NewUnverified,
            "no reference baseline for this identifier",
        ));
    }
    for id in &report.removed {
        verdict.cases.push(CaseVerdict::not_compared(
            id,
            CaseClass::Removed,
            "no current measurement for this identifier",
        ));
    }
}

/// Classify one comparable identifier.
fn classify_case(
    current_root: &Path,
    case: &ComparableCase,
    threshold: f64,
) -> Result<CaseVerdict, BaselineError> {
    let change = read_case_change(
        current_root,
        &case.current_directory.display().to_string(),
        &case.full_id,
    )?;
    let classification: Classification = change.classify(&case.full_id, threshold)?;

    Ok(CaseVerdict::classified(
        &case.full_id,
        classification.class,
        classification.lower_bound,
        classification.upper_bound,
    ))
}

/// Decide the lane outcome from the classified identifiers.
fn outcome_for(verdict: &BaselineVerdict, unreadable: usize) -> Outcome {
    if unreadable > 0 {
        Outcome::Incomplete
    } else if verdict
        .cases
        .iter()
        .any(|case| case.class == CaseClass::Regressed)
    {
        Outcome::Regression
    } else {
        Outcome::Pass
    }
}

/// Explain a non-passing outcome.
fn reason_for(verdict: &BaselineVerdict, unreadable: usize) -> Option<String> {
    if unreadable > 0 {
        return Some(format!(
            "{unreadable} comparable identifier(s) produced no readable change estimates"
        ));
    }
    let regressed = verdict.regressed_ids();
    if regressed.is_empty() {
        return None;
    }

    Some(format!(
        "{} benchmark id(s) regressed: {}",
        regressed.len(),
        regressed.join(", ")
    ))
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
