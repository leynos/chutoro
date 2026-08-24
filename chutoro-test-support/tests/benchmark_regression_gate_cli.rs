//! Behavioural tests for the benchmark regression gate binary.

use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

use chutoro_test_support::process::find_test_binary;
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
    #[from(gate_runner)] gate_runner_result: Result<GateRunner, Box<dyn Error>>,
    #[case] case: GateCase,
) {
    let gate_runner = gate_runner_result.expect("gate runner must be created");
    let output = gate_runner
        .run(case.event, case.policy)
        .expect("benchmark_regression_gate binary must run");
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
#[case("schedule", "schedule")]
#[case("workflow_dispatch", "workflow_dispatch")]
#[case("pull_request_target", "pull_request")]
#[case("push", "other")]
fn benchmark_gate_binary_reports_event(
    #[from(gate_runner)] gate_runner_result: Result<GateRunner, Box<dyn Error>>,
    #[case] event: &str,
    #[case] expected_event: &str,
) {
    let gate_runner = gate_runner_result.expect("gate runner must be created");
    let output = gate_runner
        .run(event, None)
        .expect("benchmark_regression_gate binary must run");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("expected success, got failure: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let printed_event = parse_value(&stdout, "event").unwrap_or("<missing>");
    assert_eq!(printed_event, expected_event);
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
        let binary_path = find_test_binary("benchmark_regression_gate")?;
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

fn parse_value<'a>(stdout: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix(&prefix) {
            return Some(value.trim());
        }
    }

    None
}
