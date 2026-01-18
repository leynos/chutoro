//! Behavioural tests for the nightly Kani gate binary.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
fn kani_gate_binary_outputs_decision(gate_runner: GateRunner, #[case] case: GateCase) {
    let output = gate_runner.run(case.commit_epoch, case.now_epoch, case.force);
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
fn kani_gate_binary_rejects_invalid_force(gate_runner: GateRunner, #[case] force_value: &str) {
    let output = gate_runner.run(SECONDS_PER_DAY, SECONDS_PER_DAY, Some(force_value));
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!("expected failure, got success: {stdout}");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid boolean"));
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

    fn run(&self, commit_epoch: u64, now_epoch: u64, force: Option<&str>) -> std::process::Output {
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

        command.output().expect("failed to run kani_nightly_gate")
    }
}

fn binary_path() -> PathBuf {
    if let Ok(value) = env::var("CARGO_BIN_EXE_kani_nightly_gate") {
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
    let direct = with_exe_suffix(target_dir.join("kani_nightly_gate"));
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
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if !metadata.is_file() {
            continue;
        }

        let file_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => name,
            None => continue,
        };
        if !has_expected_suffix(&path, file_name) {
            continue;
        }

        let file_stem = match path.file_stem().and_then(|stem| stem.to_str()) {
            Some(stem) => stem,
            None => continue,
        };
        if file_stem.starts_with("kani_nightly_gate") {
            return Some(path);
        }
    }

    None
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
