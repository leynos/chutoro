//! Command-line entry point for the benchmark baseline helper.
//!
//! The helper runs the three stages of the scheduled benchmark comparison
//! lane: `inventory` records what a measurement wrote, `stage` transfers
//! reference data into the current Criterion tree, and `classify` applies the
//! regression policy. Each stage maps to a documented exit code so the
//! workflow can tell a performance regression from a lane defect.

use std::ffi::OsString;
use std::io::Write;
use std::process::ExitCode;

use chutoro_test_support::ci::benchmark_baseline::args::{Command, parse};
use chutoro_test_support::ci::benchmark_baseline::runner::{
    run_classify, run_inventory, run_stage,
};
use chutoro_test_support::ci::benchmark_baseline::{
    BaselineError, BaselineVerdict, EXIT_PASS, Outcome, exit_code,
};
use mockable::{DefaultEnv, Env};

fn main() -> ExitCode {
    init_tracing();
    let arguments = collect_arguments();

    match run(&arguments) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            let code = error.exit_code();
            let _report = writeln!(std::io::stderr(), "error: {error}");
            ExitCode::from(u8::try_from(code).unwrap_or(u8::MAX))
        }
    }
}

/// Collect the process arguments after the program name.
fn collect_arguments() -> Vec<String> {
    std::env::args_os()
        .skip(1)
        .map(OsString::into_string)
        .map(Result::unwrap_or_default)
        .collect()
}

/// Dispatch one parsed command and return its exit code.
fn run(arguments: &[String]) -> Result<u8, BaselineError> {
    match parse(arguments)? {
        Command::Help => {
            let _write_result = write_usage(&mut std::io::stdout().lock());
            Ok(to_exit_code(EXIT_PASS))
        }
        Command::Inventory(args) => {
            let manifest = run_inventory(&args)?;
            let _report = writeln!(
                std::io::stdout(),
                "inventory bench={} baseline={} ids={} root={}",
                manifest.bench,
                manifest.baseline,
                manifest.cases.len(),
                manifest.output_root.display(),
            );
            Ok(to_exit_code(EXIT_PASS))
        }
        Command::Stage(args) => {
            let report = run_stage(&args)?;
            let _report = writeln!(
                std::io::stdout(),
                "stage bench={} comparable={} new_unverified={} removed={}",
                report.bench,
                report.comparable.len(),
                report.new_unverified.len(),
                report.removed.len(),
            );
            if let Some(path) = &args.filter_output {
                if let Ok(filter) = std::fs::read_to_string(path) {
                    let _filter = writeln!(std::io::stdout(), "bench_filter={}", filter.trim());
                }
            }
            Ok(to_exit_code(EXIT_PASS))
        }
        Command::Classify(args) => {
            let verdict = run_classify(&args)?;
            emit_github_output(&verdict)?;
            let _report = writeln!(std::io::stdout(), "classify {}", verdict.to_line());
            let regressed = verdict.regressed_ids();
            if !regressed.is_empty() {
                let _report = writeln!(std::io::stdout(), "regressed_ids={}", regressed.join(","));
            }
            Ok(to_exit_code(exit_code(verdict.outcome)))
        }
    }
}

/// Narrow an exit code to the process return type.
fn to_exit_code(code: i32) -> u8 {
    u8::try_from(code).unwrap_or(u8::MAX)
}

/// Write usage for a maintainer who ran the helper by hand.
fn write_usage(writer: &mut impl Write) -> std::io::Result<()> {
    writeln!(
        writer,
        "\
benchmark_baseline — Criterion baseline discovery, transfer, and policy

Usage:
  benchmark_baseline inventory --workspace DIR --bench NAME --baseline NAME
                               --revision SHA --output FILE
                               [--fresh-since FILE] [--bench-argument ARG]...
  benchmark_baseline stage --bench NAME --reference-manifest FILE
                           --current-manifest FILE --evidence DIR
                           [--filter-output FILE]
  benchmark_baseline classify --bench NAME --preflight FILE --verdict FILE
                              [--threshold FRACTION] [--no-reference REASON]

Exit codes:
  0 pass            comparison produced comparable evidence and no regression
  1 regression      a comparable benchmark id regressed
  2 incomplete      the lane could not produce comparable evidence
  3 incompatible    the two sides used different tools, targets, or settings
  4 no_reference    no previous revision existed to compare against
 64 usage           invalid arguments"
    )
}

/// Append the classification outcome and counts to `GITHUB_OUTPUT` when set.
fn emit_github_output(verdict: &BaselineVerdict) -> Result<(), BaselineError> {
    let Some(path) = DefaultEnv.string("GITHUB_OUTPUT") else {
        return Ok(());
    };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|source| BaselineError::Write {
            path: path.into(),
            source,
            remedy: "ensure GITHUB_OUTPUT names a writable file",
        })?;

    writeln!(file, "outcome={}", verdict.outcome.as_str())?;
    writeln!(file, "comparable={}", verdict.comparable_count)?;
    writeln!(file, "regressed={}", verdict.regressed_ids().len())?;
    writeln!(file, "stage={}", verdict.stage)?;
    if verdict.outcome == Outcome::Pass {
        return Ok(());
    }
    let reason = verdict.reason.clone().unwrap_or_default();
    writeln!(file, "reason={}", sanitize_output(&reason))?;

    Ok(())
}

/// Flatten a reason so it stays a single `GITHUB_OUTPUT` line.
fn sanitize_output(value: &str) -> String {
    value.replace(['\n', '\r'], " ")
}

/// Initialize non-failing stderr tracing for the helper process.
fn init_tracing() {
    let _subscriber_init_result = tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .try_init();
}

#[cfg(test)]
mod tests {
    //! Tests for helper output formatting.

    use super::sanitize_output;

    #[test]
    fn multiline_reasons_become_one_line() {
        assert_eq!(
            sanitize_output("first\nsecond\r\nthird"),
            "first second  third"
        );
    }

    #[test]
    fn single_line_reasons_are_unchanged() {
        assert_eq!(sanitize_output("one line"), "one line");
    }
}
