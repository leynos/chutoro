//! Emit a decision for whether the nightly Kani workflow should run.

use std::env::VarError;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use chutoro_test_support::ci::nightly_gate::should_run_kani_full;
use mockable::{DefaultEnv, Env};

fn main() -> Result<(), Box<dyn Error>> {
    let force = read_force_flag()?;
    let commit_epoch = read_commit_epoch()?;
    let now_epoch = read_now_epoch()?;
    let decision = should_run_kani_full(commit_epoch, now_epoch, force)?;

    let should_run = decision.should_run;
    let reason = &decision.reason;

    emit_github_output(should_run, reason)?;

    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "should_run={should_run}")?;
    writeln!(stdout, "reason={reason}")?;

    Ok(())
}

/// Read the optional override that forces the Kani job to run.
fn read_force_flag() -> Result<bool, Box<dyn Error>> {
    read_force_flag_with_env(&DefaultEnv)
}

/// Read the optional force override through an injected environment reader.
fn read_force_flag_with_env(env: &dyn Env) -> Result<bool, Box<dyn Error>> {
    let raw = read_optional_env(env, "CHUTORO_KANI_FORCE")?.unwrap_or_default();
    if raw.is_empty() {
        return Ok(false);
    }

    parse_bool(&raw).map_err(Into::into)
}

/// Parse one of the gate's accepted boolean environment value spellings.
fn parse_bool(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean value: {value}")),
    }
}

/// Read the main commit timestamp from an override or the local Git history.
fn read_commit_epoch() -> Result<u64, Box<dyn Error>> {
    if let Some(epoch) = read_commit_epoch_with_env(&DefaultEnv)? {
        return Ok(epoch);
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

/// Read the optional main-commit timestamp override through an injected reader.
fn read_commit_epoch_with_env(env: &dyn Env) -> Result<Option<u64>, Box<dyn Error>> {
    match read_optional_env(env, "CHUTORO_KANI_COMMIT_EPOCH")? {
        Some(value) => Ok(Some(value.parse::<u64>()?)),
        None => Ok(None),
    }
}

/// Read the current timestamp from an override or the system clock.
fn read_now_epoch() -> Result<u64, Box<dyn Error>> {
    if let Some(epoch) = read_now_epoch_with_env(&DefaultEnv)? {
        return Ok(epoch);
    }

    let duration = SystemTime::now().duration_since(UNIX_EPOCH)?;
    Ok(duration.as_secs())
}

/// Read the optional current-timestamp override through an injected reader.
fn read_now_epoch_with_env(env: &dyn Env) -> Result<Option<u64>, Box<dyn Error>> {
    match read_optional_env(env, "CHUTORO_KANI_NOW_EPOCH")? {
        Some(value) => Ok(Some(value.parse::<u64>()?)),
        None => Ok(None),
    }
}

/// Append the Kani decision to GitHub's workflow output file.
fn emit_github_output(should_run: bool, reason: &str) -> Result<(), Box<dyn Error>> {
    let output_path = read_optional_env(&DefaultEnv, "GITHUB_OUTPUT")?.unwrap_or_default();
    if output_path.is_empty() {
        return Ok(());
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;

    writeln!(file, "should_run={should_run}")?;
    write_github_output_value(&mut file, "reason", reason)?;

    Ok(())
}

/// Write one GitHub output value using a delimiter when it contains newlines.
fn write_github_output_value(
    file: &mut impl Write,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn Error>> {
    if !value.contains('\n') && !value.contains('\r') {
        writeln!(file, "{key}={value}")?;
        return Ok(());
    }

    let delimiter = "CHUTORO_EOF";
    if value.contains(delimiter) {
        return Err(format!("output value for {key} contains {delimiter}").into());
    }

    writeln!(file, "{key}<<{delimiter}")?;
    writeln!(file, "{value}")?;
    writeln!(file, "{delimiter}")?;

    Ok(())
}

/// Return the optional environment value named `name`.
fn read_optional_env(env: &dyn Env, name: &str) -> Result<Option<String>, Box<dyn Error>> {
    match env.raw(name) {
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    //! Tests for Kani gate environment handling.

    use std::{env::VarError, ffi::OsString};

    use mockable::MockEnv;

    use super::{
        read_commit_epoch_with_env, read_force_flag_with_env, read_now_epoch_with_env,
        read_optional_env,
    };

    #[test]
    fn optional_environment_returns_present_value() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "CHUTORO_KANI_FORCE");
            Ok("true".to_owned())
        });

        assert_eq!(
            read_optional_env(&env, "CHUTORO_KANI_FORCE").expect("environment read must succeed"),
            Some("true".to_owned())
        );
    }

    #[test]
    fn optional_environment_maps_not_present_to_none() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "CHUTORO_KANI_FORCE");
            Err(VarError::NotPresent)
        });

        assert_eq!(
            read_optional_env(&env, "CHUTORO_KANI_FORCE").expect("environment read must succeed"),
            None
        );
    }

    #[test]
    fn optional_environment_returns_non_not_present_errors() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "CHUTORO_KANI_FORCE");
            Err(VarError::NotUnicode(OsString::from("invalid")))
        });

        assert!(read_optional_env(&env, "CHUTORO_KANI_FORCE").is_err());
    }

    #[test]
    fn force_flag_defaults_to_false_when_unset() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "CHUTORO_KANI_FORCE");
            Err(VarError::NotPresent)
        });

        assert!(!read_force_flag_with_env(&env).expect("force flag must parse"));
    }

    #[test]
    fn force_flag_accepts_valid_value() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "CHUTORO_KANI_FORCE");
            Ok("true".to_owned())
        });

        assert!(read_force_flag_with_env(&env).expect("force flag must parse"));
    }

    #[test]
    fn force_flag_rejects_invalid_value() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "CHUTORO_KANI_FORCE");
            Ok("maybe".to_owned())
        });

        assert!(read_force_flag_with_env(&env).is_err());
    }

    #[test]
    fn commit_epoch_uses_environment_override() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "CHUTORO_KANI_COMMIT_EPOCH");
            Ok("1725000000".to_owned())
        });

        assert_eq!(
            read_commit_epoch_with_env(&env).expect("commit epoch must parse"),
            Some(1_725_000_000)
        );
    }

    #[test]
    fn now_epoch_uses_environment_override() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "CHUTORO_KANI_NOW_EPOCH");
            Ok("1725000001".to_owned())
        });

        assert_eq!(
            read_now_epoch_with_env(&env).expect("current epoch must parse"),
            Some(1_725_000_001)
        );
    }
}
