//! Emit benchmark regression mode for CI workflows.

use std::error::Error;

use chutoro_test_support::ci::benchmark_regression_profile::{
    BenchmarkCiPolicy, BenchmarkRegressionProfile,
};
use chutoro_test_support::github_output::{self, OutputPair};

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

    println!("mode={}", mode.as_str());
    println!("should_compare={should_compare}");
    println!("event={}", event.as_str());
    println!("policy={}", policy.as_str());
    println!("reason={reason}");

    Ok(())
}

fn init_tracing() {
    let _subscriber_init_result = tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .try_init();
}

fn emit_github_output(
    profile: BenchmarkRegressionProfile,
    reason: &str,
) -> Result<(), Box<dyn Error>> {
    let mode = profile.mode();
    let should_compare = mode.should_compare().to_string();
    github_output::append_if_configured(&[
        OutputPair::new("mode", mode.as_str()),
        OutputPair::new("should_compare", &should_compare),
        OutputPair::new("event", profile.event().as_str()),
        OutputPair::new("policy", profile.policy().as_str()),
        OutputPair::new("reason", reason),
    ])?;
    Ok(())
}
