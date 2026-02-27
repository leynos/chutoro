//! Behavioural tests for the benchmark regression gate binary.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
fn benchmark_gate_binary_outputs_expected_mode(gate_runner: GateRunner, #[case] case: GateCase) {
    let output = gate_runner.run(case.event, case.policy);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("expected success, got failure: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mode = match parse_value(&stdout, "mode") {
        Some(value) => value,
        None => panic!("missing mode output: {stdout}"),
    };
    let should_compare = match parse_value(&stdout, "should_compare") {
        Some("true") => true,
        Some("false") => false,
        Some(other) => panic!("unexpected should_compare value: {other}"),
        None => panic!("missing should_compare output: {stdout}"),
    };

    assert_eq!(mode, case.expected_mode);
    assert_eq!(should_compare, case.expected_should_compare);
}

#[rstest]
#[case("schedule")]
#[case("push")]
fn benchmark_gate_binary_reports_event(gate_runner: GateRunner, #[case] event: &str) {
    let output = gate_runner.run(event, None);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("expected success, got failure: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let printed_event = parse_value(&stdout, "event").unwrap_or("<missing>");
    assert!(!printed_event.is_empty());
}

#[fixture]
fn gate_runner() -> GateRunner {
    GateRunner::new()
}

struct GateRunner {
    binary_path: PathBuf,
}

impl GateRunner {
    fn new() -> Self {
        Self {
            binary_path: binary_path(),
        }
    }

    fn run(&self, event: &str, policy: Option<&str>) -> std::process::Output {
        let mut command = Command::new(&self.binary_path);
        command
            .env("GITHUB_EVENT_NAME", event)
            .env_remove("GITHUB_OUTPUT")
            .env_remove("CHUTORO_BENCH_CI_POLICY");

        if let Some(value) = policy {
            command.env("CHUTORO_BENCH_CI_POLICY", value);
        }

        command
            .output()
            .expect("failed to run benchmark_regression_gate")
    }
}

fn binary_path() -> PathBuf {
    if let Ok(value) = env::var("CARGO_BIN_EXE_benchmark_regression_gate") {
        return with_exe_suffix(PathBuf::from(value));
    }

    let current_exe = match env::current_exe() {
        Ok(path) => path,
        Err(error) => panic!("failed to locate current test binary: {error}"),
    };
    let deps_dir = match current_exe.parent() {
        Some(dir) => dir.to_path_buf(),
        None => panic!("failed to resolve deps directory from test binary"),
    };
    let target_dir = match deps_dir.parent() {
        Some(dir) => dir.to_path_buf(),
        None => panic!("failed to resolve target directory from deps"),
    };
    let direct = with_exe_suffix(target_dir.join("benchmark_regression_gate"));
    if direct.exists() {
        return direct;
    }

    match find_in_deps(&deps_dir) {
        Some(path) => path,
        None => panic!("failed to locate benchmark_regression_gate binary"),
    }
}

fn find_in_deps(deps_dir: &Path) -> Option<PathBuf> {
    fs::read_dir(deps_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .find_map(is_matching_binary)
}

fn is_matching_binary(entry: fs::DirEntry) -> Option<PathBuf> {
    let path = entry.path();
    let metadata = entry.metadata().ok()?;

    if !metadata.is_file() {
        return None;
    }

    let file_name = path.file_name()?.to_str()?;
    if !has_expected_suffix(&path, file_name) {
        return None;
    }

    let file_stem = path.file_stem()?.to_str()?;
    if file_stem.starts_with("benchmark_regression_gate") {
        Some(path)
    } else {
        None
    }
}

fn has_expected_suffix(path: &Path, file_name: &str) -> bool {
    let suffix = env::consts::EXE_SUFFIX;
    if suffix.is_empty() {
        return path.extension().is_none();
    }

    file_name.ends_with(suffix)
}

fn with_exe_suffix(mut path: PathBuf) -> PathBuf {
    let suffix = env::consts::EXE_SUFFIX;
    if suffix.is_empty() {
        return path;
    }

    let file_name = match path.file_name().and_then(|name| name.to_str()) {
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
