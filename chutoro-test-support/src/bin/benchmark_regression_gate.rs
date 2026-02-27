//! Emit benchmark regression mode for CI workflows.

use std::env;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;

use chutoro_test_support::ci::benchmark_regression_profile::{
    BenchmarkCiPolicy, BenchmarkRegressionProfile,
};

fn main() -> Result<(), Box<dyn Error>> {
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

    println!("mode={}", mode.as_str());
    println!("should_compare={should_compare}");
    println!("event={}", event.as_str());
    println!("policy={}", policy.as_str());
    println!("reason={reason}");

    Ok(())
}

fn emit_github_output(
    profile: BenchmarkRegressionProfile,
    reason: &str,
) -> Result<(), Box<dyn Error>> {
    let output_path = read_optional_env("GITHUB_OUTPUT")?.unwrap_or_default();
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

fn read_optional_env(name: &str) -> Result<Option<String>, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}
