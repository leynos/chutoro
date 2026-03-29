//! Behavioural tests for the benchmark regression gate binary.

use anyhow::Context;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

use camino::{Utf8Path, Utf8PathBuf};
use cap_std::{ambient_authority, fs_utf8::Dir};
use rstest::{fixture, rstest};

#[derive(Debug, Clone, Copy)]
struct GateCase {
    event: &'static str,
    policy: Option<&'static str>,
    expected_mode: &'static str,
    expected_should_compare: bool,
}

#[rstest]
#[case::pr_default(GateCase {
    event: "pull_request",
    policy: None,
    expected_mode: "discovery_only",
    expected_should_compare: false,
})]
#[case::scheduled_default(GateCase {
    event: "schedule",
    policy: None,
    expected_mode: "baseline_compare",
    expected_should_compare: true,
})]
#[case::workflow_dispatch_default(GateCase {
    event: "workflow_dispatch",
    policy: None,
    expected_mode: "baseline_compare",
    expected_should_compare: true,
})]
#[case::pr_always_baseline(GateCase {
    event: "pull_request",
    policy: Some("always-baseline"),
    expected_mode: "baseline_compare",
    expected_should_compare: true,
})]
#[case::schedule_disabled(GateCase {
    event: "schedule",
    policy: Some("disabled"),
    expected_mode: "disabled",
    expected_should_compare: false,
})]
#[case::invalid_policy_falls_back_to_default(GateCase {
    event: "pull_request",
    policy: Some("invalid-policy"),
    expected_mode: "discovery_only",
    expected_should_compare: false,
})]
fn benchmark_gate_binary_outputs_expected_mode(
    gate_runner: Result<GateRunner, Box<dyn Error>>,
    #[case] case: GateCase,
) -> Result<(), Box<dyn Error>> {
    let gate_runner = gate_runner?;
    let output = gate_runner.run(case.event, case.policy)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(
            std::io::Error::other(format!("expected success, got failure: {stderr}",)).into(),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mode = parse_value(&stdout, "mode")
        .ok_or_else(|| std::io::Error::other(format!("missing mode output: {stdout}")))?;
    let should_compare = match parse_value(&stdout, "should_compare") {
        Some("true") => true,
        Some("false") => false,
        Some(other) => {
            return Err(std::io::Error::other(
                format!("unexpected should_compare value: {other}",),
            )
            .into());
        }
        None => {
            return Err(
                std::io::Error::other(format!("missing should_compare output: {stdout}",)).into(),
            );
        }
    };

    assert_eq!(mode, case.expected_mode);
    assert_eq!(should_compare, case.expected_should_compare);
    Ok(())
}

#[rstest]
#[case("schedule", "schedule")]
#[case("workflow_dispatch", "workflow_dispatch")]
#[case("pull_request_target", "pull_request")]
#[case("push", "other")]
fn benchmark_gate_binary_reports_event(
    gate_runner: Result<GateRunner, Box<dyn Error>>,
    #[case] event: &str,
    #[case] expected_event: &str,
) -> Result<(), Box<dyn Error>> {
    let gate_runner = gate_runner?;
    let output = gate_runner.run(event, None)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(
            std::io::Error::other(format!("expected success, got failure: {stderr}",)).into(),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let printed_event = parse_value(&stdout, "event").unwrap_or("<missing>");
    assert_eq!(printed_event, expected_event);
    Ok(())
}

#[fixture]
fn gate_runner() -> Result<GateRunner, Box<dyn Error>> {
    GateRunner::new()
}

struct GateRunner {
    binary_path: PathBuf,
}

impl GateRunner {
    fn new() -> Result<Self, Box<dyn Error>> {
        let binary_path = binary_path().map_err(std::io::Error::other)?;
        Ok(Self { binary_path })
    }

    fn run(&self, event: &str, policy: Option<&str>) -> std::io::Result<std::process::Output> {
        let mut command = Command::new(&self.binary_path);
        command
            .env("GITHUB_EVENT_NAME", event)
            .env_remove("GITHUB_OUTPUT")
            .env_remove("CHUTORO_BENCH_CI_POLICY");

        if let Some(value) = policy {
            command.env("CHUTORO_BENCH_CI_POLICY", value);
        }

        command.output()
    }
}

fn binary_path() -> anyhow::Result<PathBuf> {
    if let Ok(value) = env::var("CARGO_BIN_EXE_benchmark_regression_gate") {
        return Ok(with_exe_suffix(Utf8PathBuf::from(value)).into_std_path_buf());
    }

    let current_exe = env::current_exe().context("failed to locate current test binary")?;
    let current_exe = Utf8PathBuf::from_path_buf(current_exe)
        .map_err(|path| anyhow::anyhow!("test binary path must be UTF-8: {}", path.display()))?;
    let deps_dir = current_exe
        .parent()
        .map(Utf8Path::to_path_buf)
        .context("failed to resolve deps directory from test binary")?;
    let target_dir = deps_dir
        .parent()
        .map(Utf8Path::to_path_buf)
        .context("failed to resolve target directory from deps")?;
    let direct = with_exe_suffix(target_dir.join("benchmark_regression_gate"));
    if direct.exists() {
        return Ok(direct.into_std_path_buf());
    }

    find_in_deps(&deps_dir)?
        .map(Utf8PathBuf::into_std_path_buf)
        .context("failed to locate benchmark_regression_gate binary")
}

fn find_in_deps(deps_dir: &Utf8Path) -> anyhow::Result<Option<Utf8PathBuf>> {
    let dir = Dir::open_ambient_dir(deps_dir, ambient_authority())
        .with_context(|| format!("failed to open deps directory `{deps_dir}`"))?;
    let mut entries = dir
        .entries()
        .with_context(|| format!("failed to enumerate deps directory `{deps_dir}`"))?;
    entries.try_fold(None, |found, entry| {
        if found.is_some() {
            return Ok(found);
        }

        let entry = entry
            .with_context(|| format!("failed to read entry in deps directory `{deps_dir}`"))?;
        is_matching_binary(deps_dir, entry)
    })
}

fn is_matching_binary(
    deps_dir: &Utf8Path,
    entry: cap_std::fs_utf8::DirEntry,
) -> anyhow::Result<Option<Utf8PathBuf>> {
    let file_name = entry
        .file_name()
        .with_context(|| format!("failed to read file name in `{deps_dir}`"))?;
    let metadata = entry
        .metadata()
        .with_context(|| format!("failed to read metadata for `{file_name}` in `{deps_dir}`"))?;

    if !metadata.is_file() {
        return Ok(None);
    }

    if !has_expected_suffix(&file_name) {
        return Ok(None);
    }

    let Some(file_stem) = Utf8Path::new(&file_name).file_stem() else {
        return Ok(None);
    };
    if file_stem == "benchmark_regression_gate"
        || file_stem.starts_with("benchmark_regression_gate-")
    {
        Ok(Some(deps_dir.join(file_name)))
    } else {
        Ok(None)
    }
}

fn has_expected_suffix(file_name: &str) -> bool {
    let suffix = env::consts::EXE_SUFFIX;
    if suffix.is_empty() {
        return Utf8Path::new(file_name).extension().is_none();
    }

    file_name.ends_with(suffix)
}

fn with_exe_suffix(mut path: Utf8PathBuf) -> Utf8PathBuf {
    let suffix = env::consts::EXE_SUFFIX;
    if suffix.is_empty() {
        return path;
    }

    let file_name = match path.file_name() {
        Some(name) => name,
        None => return path,
    };
    if file_name.ends_with(suffix) {
        return path;
    }

    let updated = format!("{file_name}{suffix}");
    path.set_file_name(updated);
    path
}

fn parse_value<'a>(stdout: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix(&prefix) {
            return Some(value.trim());
        }
    }

    None
}
