//! Emit benchmark regression mode for CI workflows.

use std::env::VarError;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;

use chutoro_test_support::ci::benchmark_regression_profile::{
    BenchmarkCiPolicy, BenchmarkRegressionProfile,
};
use mockable::{DefaultEnv, Env};

fn main() -> Result<(), Box<dyn Error>> {
    init_tracing();

    let profile = BenchmarkRegressionProfile::load(BenchmarkCiPolicy::ScheduledBaseline);
    let mode = profile.mode();
    let should_compare = mode.should_compare();
    let event = profile.event();
    let policy = profile.policy();
    let reason = format!(
        "event={} policy={} mode={}",
        event.as_str(),
        policy.as_str(),
        mode.as_str(),
    );

    emit_github_output(profile, &reason)?;

    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "mode={}", mode.as_str())?;
    writeln!(stdout, "should_compare={should_compare}")?;
    writeln!(stdout, "event={}", event.as_str())?;
    writeln!(stdout, "policy={}", policy.as_str())?;
    writeln!(stdout, "reason={reason}")?;

    Ok(())
}

/// Initialise non-failing stderr tracing for the gate process.
fn init_tracing() {
    let _subscriber_init_result = tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .try_init();
}

/// Append the resolved benchmark profile to GitHub's workflow output file.
fn emit_github_output(
    profile: BenchmarkRegressionProfile,
    reason: &str,
) -> Result<(), Box<dyn Error>> {
    let output_path = read_optional_env(&DefaultEnv, "GITHUB_OUTPUT")?.unwrap_or_default();
    if output_path.is_empty() {
        return Ok(());
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;

    let mode = profile.mode();
    writeln!(file, "mode={}", mode.as_str())?;
    writeln!(file, "should_compare={}", mode.should_compare())?;
    writeln!(file, "event={}", profile.event().as_str())?;
    writeln!(file, "policy={}", profile.policy().as_str())?;
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
    //! Tests for benchmark-regression gate environment handling.

    use std::{env::VarError, ffi::OsString};

    use mockable::MockEnv;

    use super::read_optional_env;

    #[test]
    fn optional_environment_returns_present_value() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "GITHUB_OUTPUT");
            Ok("output.txt".to_owned())
        });

        assert_eq!(
            read_optional_env(&env, "GITHUB_OUTPUT").expect("environment read must succeed"),
            Some("output.txt".to_owned())
        );
    }

    #[test]
    fn optional_environment_maps_not_present_to_none() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "GITHUB_OUTPUT");
            Err(VarError::NotPresent)
        });

        assert_eq!(
            read_optional_env(&env, "GITHUB_OUTPUT").expect("environment read must succeed"),
            None
        );
    }

    #[test]
    fn optional_environment_returns_non_not_present_errors() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, "GITHUB_OUTPUT");
            Err(VarError::NotUnicode(OsString::from("invalid")))
        });

        assert!(read_optional_env(&env, "GITHUB_OUTPUT").is_err());
    }
}
