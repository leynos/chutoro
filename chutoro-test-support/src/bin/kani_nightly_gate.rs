//! Emit a decision for whether the nightly Kani workflow should run.

use std::env;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use chutoro_test_support::ci::nightly_gate::should_run_kani_full;

fn main() -> Result<(), Box<dyn Error>> {
    let force = read_force_flag()?;
    let commit_epoch = read_commit_epoch()?;
    let now_epoch = current_epoch()?;
    let decision = should_run_kani_full(commit_epoch, now_epoch, force)?;

    emit_github_output(decision.should_run, &decision.reason)?;

    println!("should_run={}", decision.should_run);
    println!("reason={}", decision.reason);

    Ok(())
}

fn read_force_flag() -> Result<bool, Box<dyn Error>> {
    let raw = match env::var("CHUTORO_KANI_FORCE") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => String::new(),
        Err(error) => return Err(error.into()),
    };
    if raw.is_empty() {
        return Ok(false);
    }

    parse_bool(&raw).map_err(|message| message.into())
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean value: {value}")),
    }
}

fn read_commit_epoch() -> Result<u64, Box<dyn Error>> {
    let output = Command::new("git")
        .args(["log", "-1", "--format=%ct"])
        .output()?;

    if !output.status.success() {
        let status = output.status;
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git log failed with {status}: {stderr}").into());
    }

    let stdout = String::from_utf8(output.stdout)?;
    let trimmed = stdout.trim();

    Ok(trimmed.parse::<u64>()?)
}

fn current_epoch() -> Result<u64, Box<dyn Error>> {
    let duration = SystemTime::now().duration_since(UNIX_EPOCH)?;
    Ok(duration.as_secs())
}

fn emit_github_output(should_run: bool, reason: &str) -> Result<(), Box<dyn Error>> {
    let output_path = match env::var("GITHUB_OUTPUT") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => String::new(),
        Err(error) => return Err(error.into()),
    };
    if output_path.is_empty() {
        return Ok(());
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;

    writeln!(file, "should_run={should_run}")?;
    writeln!(file, "reason={reason}")?;

    Ok(())
}
