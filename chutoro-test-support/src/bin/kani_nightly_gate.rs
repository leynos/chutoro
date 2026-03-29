//! Emit a decision for whether the nightly Kani workflow should run.

use std::env;
use std::error::Error;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use chutoro_test_support::ci::nightly_gate::should_run_kani_full;
use chutoro_test_support::github_output::{self, OutputPair};

fn main() -> Result<(), Box<dyn Error>> {
    let force = read_force_flag()?;
    let commit_epoch = read_commit_epoch()?;
    let now_epoch = read_now_epoch()?;
    let decision = should_run_kani_full(commit_epoch, now_epoch, force)?;

    let should_run = decision.should_run;
    let reason = &decision.reason;

    emit_github_output(should_run, reason)?;

    println!("should_run={should_run}");
    println!("reason={reason}");

    Ok(())
}

fn read_force_flag() -> Result<bool, Box<dyn Error>> {
    let raw = read_optional_env("CHUTORO_KANI_FORCE")?.unwrap_or_default();
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
    if let Some(value) = read_optional_env("CHUTORO_KANI_COMMIT_EPOCH")? {
        return Ok(value.parse::<u64>()?);
    }

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

fn read_now_epoch() -> Result<u64, Box<dyn Error>> {
    if let Some(value) = read_optional_env("CHUTORO_KANI_NOW_EPOCH")? {
        return Ok(value.parse::<u64>()?);
    }

    let duration = SystemTime::now().duration_since(UNIX_EPOCH)?;
    Ok(duration.as_secs())
}

fn emit_github_output(should_run: bool, reason: &str) -> Result<(), Box<dyn Error>> {
    let should_run = should_run.to_string();
    github_output::append_if_configured(&[
        OutputPair::new("should_run", &should_run),
        OutputPair::new("reason", reason),
    ])?;
    Ok(())
}

fn read_optional_env(name: &str) -> Result<Option<String>, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}
