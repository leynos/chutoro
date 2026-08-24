//! Behavioural tests for the nightly Kani gate binary.

use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

use chutoro_test_support::process::find_test_binary;
use rstest::{fixture, rstest};

const SECONDS_PER_DAY: u64 = 86_400;

#[derive(Debug, Clone, Copy)]
struct GateCase {
    commit_epoch: u64,
    now_epoch: u64,
    force: Option<&'static str>,
    expected_run: bool,
}

#[rstest]
#[case::same_timestamp(GateCase {
    commit_epoch: SECONDS_PER_DAY,
    now_epoch: SECONDS_PER_DAY,
    force: None,
    expected_run: true,
})]
#[case::within_window(GateCase {
    commit_epoch: SECONDS_PER_DAY,
    now_epoch: SECONDS_PER_DAY * 2,
    force: None,
    expected_run: true,
})]
#[case::stale_commit(GateCase {
    commit_epoch: SECONDS_PER_DAY - 1,
    now_epoch: SECONDS_PER_DAY * 2 + 1,
    force: None,
    expected_run: false,
})]
#[case::small_future_skew(GateCase {
    commit_epoch: SECONDS_PER_DAY + 10,
    now_epoch: SECONDS_PER_DAY + 9,
    force: None,
    expected_run: false,
})]
#[case::force_override(GateCase {
    commit_epoch: SECONDS_PER_DAY + 10,
    now_epoch: SECONDS_PER_DAY + 9,
    force: Some("true"),
    expected_run: true,
})]
fn kani_gate_binary_outputs_decision(
    #[from(gate_runner)] gate_runner_result: Result<GateRunner, Box<dyn Error>>,
    #[case] case: GateCase,
) {
    let gate_runner = gate_runner_result.expect("gate runner must be created");
    let output = gate_runner
        .run(case.commit_epoch, case.now_epoch, case.force)
        .expect("kani_nightly_gate binary must run");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("expected success, got failure: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let should_run = match parse_should_run(&stdout) {
        Some(value) => value,
        None => panic!("missing should_run output: {stdout}"),
    };

    assert_eq!(should_run, case.expected_run);
}

#[rstest]
#[case::invalid_force("maybe")]
#[case::invalid_force_whitespace("true-ish")]
fn kani_gate_binary_rejects_invalid_force(
    #[from(gate_runner)] gate_runner_result: Result<GateRunner, Box<dyn Error>>,
    #[case] force_value: &str,
) {
    let gate_runner = gate_runner_result.expect("gate runner must be created");
    let output = gate_runner
        .run(SECONDS_PER_DAY, SECONDS_PER_DAY, Some(force_value))
        .expect("kani_nightly_gate binary must run");
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!("expected failure, got success: {stdout}");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid boolean"));
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
        let binary_path = find_test_binary("kani_nightly_gate")?;
        Ok(Self { binary_path })
    }

    fn run(
        &self,
        commit_epoch: u64,
        now_epoch: u64,
        force: Option<&str>,
    ) -> std::io::Result<std::process::Output> {
        let mut command = Command::new(&self.binary_path);
        command
            .env("CHUTORO_KANI_COMMIT_EPOCH", commit_epoch.to_string())
            .env("CHUTORO_KANI_NOW_EPOCH", now_epoch.to_string())
            .env_remove("GITHUB_OUTPUT");

        match force {
            Some(value) => {
                command.env("CHUTORO_KANI_FORCE", value);
            }
            None => {
                command.env_remove("CHUTORO_KANI_FORCE");
            }
        }

        command.output()
    }
}

fn parse_should_run(stdout: &str) -> Option<bool> {
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix("should_run=") {
            return match value.trim() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            };
        }
    }

    None
}
