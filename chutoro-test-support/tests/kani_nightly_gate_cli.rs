//! Behavioural tests for the nightly Kani gate binary.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rstest::rstest;

const SECONDS_PER_DAY: u64 = 86_400;

#[rstest]
#[case::same_timestamp(SECONDS_PER_DAY, SECONDS_PER_DAY, None, true)]
#[case::within_window(SECONDS_PER_DAY, SECONDS_PER_DAY * 2, None, true)]
#[case::stale_commit(SECONDS_PER_DAY - 1, SECONDS_PER_DAY * 2 + 1, None, false)]
#[case::small_future_skew(SECONDS_PER_DAY + 10, SECONDS_PER_DAY + 9, None, false)]
#[case::force_override(SECONDS_PER_DAY + 10, SECONDS_PER_DAY + 9, Some("true"), true)]
fn kani_gate_binary_outputs_decision(
    #[case] commit_epoch: u64,
    #[case] now_epoch: u64,
    #[case] force: Option<&str>,
    #[case] expected_run: bool,
) {
    let output = run_gate(commit_epoch, now_epoch, force);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("expected success, got failure: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let should_run = match parse_should_run(&stdout) {
        Some(value) => value,
        None => panic!("missing should_run output: {stdout}"),
    };

    assert_eq!(should_run, expected_run);
}

#[rstest]
#[case::invalid_force("maybe")]
#[case::invalid_force_whitespace("true-ish")]
fn kani_gate_binary_rejects_invalid_force(#[case] force_value: &str) {
    let output = run_gate(SECONDS_PER_DAY, SECONDS_PER_DAY, Some(force_value));
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!("expected failure, got success: {stdout}");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid boolean"));
}

fn run_gate(commit_epoch: u64, now_epoch: u64, force: Option<&str>) -> std::process::Output {
    let mut command = Command::new(binary_path());
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

    match command.output() {
        Ok(output) => output,
        Err(error) => panic!("failed to run kani_nightly_gate: {error}"),
    }
}

fn binary_path() -> PathBuf {
    if let Ok(value) = env::var("CARGO_BIN_EXE_kani_nightly_gate") {
        return PathBuf::from(value);
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
    let direct = target_dir.join("kani_nightly_gate");
    if direct.exists() {
        return direct;
    }

    match find_in_deps(&deps_dir) {
        Some(path) => path,
        None => panic!("failed to locate kani_nightly_gate binary"),
    }
}

fn find_in_deps(deps_dir: &Path) -> Option<PathBuf> {
    let entries = match fs::read_dir(deps_dir) {
        Ok(entries) => entries,
        Err(_) => return None,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => name,
            None => continue,
        };
        if file_name.starts_with("kani_nightly_gate") {
            return Some(path);
        }
    }

    None
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
