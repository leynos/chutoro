//! Classification stage: apply the regression policy and write the verdict.

use std::path::Path;

use mockable::Env;

use super::super::args::{ClassifyArgs, NoReferenceReason};
use super::super::classification::{Classification, read_case_change};
use super::super::preflight::{ComparableCase, PreflightReport};
use super::super::verdict::{BaselineVerdict, CaseClass, CaseVerdict, NoComparison};
use super::super::{BaselineDiagnostic, BaselineError, Outcome};

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
///     failed_stage: None,
///     failure_reason: None,
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
    verdict.diagnostics.push(BaselineDiagnostic {
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

    let env = mockable::DefaultEnv;
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
    if report.bench != args.bench {
        return Err(BaselineError::Usage {
            message: format!(
                "preflight report is for benchmark target `{}`, not `{}`",
                report.bench, args.bench
            ),
            remedy: "pass the same --bench to stage and classify",
        });
    }
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
    let mut incompatible = false;
    for case in &report.comparable {
        match classify_case(&current_root, case, args.threshold) {
            Ok(case_verdict) => verdict.cases.push(case_verdict),
            Err(error) => {
                // An incompatible confidence level is a different failure
                // from unreadable evidence, and carries its own outcome.
                if error.outcome() == Some(Outcome::Incompatible) {
                    incompatible = true;
                } else {
                    unreadable += 1;
                }
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
    verdict.outcome = outcome_for(&verdict, unreadable, incompatible);
    verdict.reason = reason_for(&verdict, unreadable, incompatible);

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
fn outcome_for(verdict: &BaselineVerdict, unreadable: usize, incompatible: bool) -> Outcome {
    if incompatible {
        Outcome::Incompatible
    } else if unreadable > 0 {
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
fn reason_for(verdict: &BaselineVerdict, unreadable: usize, incompatible: bool) -> Option<String> {
    if incompatible {
        return Some(
            "at least one change interval used a confidence level other than 0.95".to_owned(),
        );
    }
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
