//! Benchmark regression profile parsing for CI and local overrides.
//!
//! This module centralizes event and policy parsing for benchmark regression
//! checks so workflows can share one deterministic decision surface.

use std::env;

/// Environment variable controlling benchmark CI policy.
pub const CHUTORO_BENCH_CI_POLICY_ENV_KEY: &str = "CHUTORO_BENCH_CI_POLICY";

/// Environment variable exposing the GitHub Actions event name.
pub const GITHUB_EVENT_NAME_ENV_KEY: &str = "GITHUB_EVENT_NAME";

/// Policy controlling when benchmark baseline comparison should run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkCiPolicy {
    /// Never run benchmark CI checks.
    Disabled,
    /// Run lightweight benchmark discovery in PRs and baseline comparison on
    /// scheduled or manual workflows.
    ScheduledBaseline,
    /// Run baseline comparison for all events.
    AlwaysBaseline,
}

impl BenchmarkCiPolicy {
    /// Returns a stable lowercase identifier for logs and workflow outputs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::ScheduledBaseline => "scheduled-baseline",
            Self::AlwaysBaseline => "always-baseline",
        }
    }
}

/// Event category used by benchmark CI policy resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkCiEvent {
    /// Pull request events.
    PullRequest,
    /// Scheduled workflow events.
    Schedule,
    /// Manual workflow dispatch events.
    WorkflowDispatch,
    /// Any event not handled explicitly.
    Other,
}

impl BenchmarkCiEvent {
    /// Returns a stable lowercase identifier for logs and workflow outputs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PullRequest => "pull_request",
            Self::Schedule => "schedule",
            Self::WorkflowDispatch => "workflow_dispatch",
            Self::Other => "other",
        }
    }
}

/// Benchmark regression execution mode resolved from event and policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkRegressionMode {
    /// Disable benchmark checks.
    Disabled,
    /// Run benchmark discovery only (`--list`) without baseline comparison.
    DiscoveryOnly,
    /// Run full Criterion baseline comparison.
    BaselineCompare,
}

impl BenchmarkRegressionMode {
    /// Returns a stable lowercase identifier for logs and workflow outputs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::DiscoveryOnly => "discovery_only",
            Self::BaselineCompare => "baseline_compare",
        }
    }

    /// Returns `true` when baseline comparison should run.
    #[must_use]
    pub const fn should_compare(self) -> bool {
        matches!(self, Self::BaselineCompare)
    }
}

/// Fully resolved benchmark CI profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkRegressionProfile {
    policy: BenchmarkCiPolicy,
    event: BenchmarkCiEvent,
    mode: BenchmarkRegressionMode,
}

impl BenchmarkRegressionProfile {
    /// Loads benchmark regression profile from environment variables.
    ///
    /// Invalid policy values produce a warning and fall back to
    /// `default_policy`.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_regression_profile::{
    ///     BenchmarkCiPolicy, BenchmarkRegressionProfile,
    /// };
    ///
    /// let profile = BenchmarkRegressionProfile::load(
    ///     BenchmarkCiPolicy::ScheduledBaseline,
    /// );
    /// assert!(!profile.mode().as_str().is_empty());
    /// ```
    #[must_use]
    pub fn load(default_policy: BenchmarkCiPolicy) -> Self {
        Self::load_with_lookup(default_policy, |key| env::var(key).ok())
    }

    fn load_with_lookup<F>(default_policy: BenchmarkCiPolicy, lookup: F) -> Self
    where
        F: Fn(&'static str) -> Option<String>,
    {
        let policy = match lookup(CHUTORO_BENCH_CI_POLICY_ENV_KEY) {
            Some(raw) => match parse_policy(&raw) {
                Ok(policy) => policy,
                Err(reason) => {
                    tracing::warn!(
                        env = CHUTORO_BENCH_CI_POLICY_ENV_KEY,
                        raw = %raw,
                        reason = %reason,
                        fallback_policy = default_policy.as_str(),
                        "invalid benchmark CI policy override; using default",
                    );
                    default_policy
                }
            },
            None => default_policy,
        };

        let event = lookup(GITHUB_EVENT_NAME_ENV_KEY)
            .as_deref()
            .map_or(BenchmarkCiEvent::Other, parse_event_name);
        let mode = resolve_regression_mode(event, policy);

        Self {
            policy,
            event,
            mode,
        }
    }

    /// Returns the parsed benchmark CI policy.
    #[must_use]
    pub const fn policy(self) -> BenchmarkCiPolicy {
        self.policy
    }

    /// Returns the parsed CI event category.
    #[must_use]
    pub const fn event(self) -> BenchmarkCiEvent {
        self.event
    }

    /// Returns the resolved benchmark regression mode.
    #[must_use]
    pub const fn mode(self) -> BenchmarkRegressionMode {
        self.mode
    }
}

/// Resolves benchmark regression mode for a CI event and policy.
#[must_use]
pub const fn resolve_regression_mode(
    event: BenchmarkCiEvent,
    policy: BenchmarkCiPolicy,
) -> BenchmarkRegressionMode {
    match policy {
        BenchmarkCiPolicy::Disabled => BenchmarkRegressionMode::Disabled,
        BenchmarkCiPolicy::AlwaysBaseline => BenchmarkRegressionMode::BaselineCompare,
        BenchmarkCiPolicy::ScheduledBaseline => {
            if matches!(
                event,
                BenchmarkCiEvent::Schedule | BenchmarkCiEvent::WorkflowDispatch
            ) {
                BenchmarkRegressionMode::BaselineCompare
            } else {
                BenchmarkRegressionMode::DiscoveryOnly
            }
        }
    }
}

/// Parses a benchmark CI policy string.
///
/// Accepted values are case-insensitive and tolerate either hyphen or
/// underscore separators.
pub fn parse_policy(raw: &str) -> Result<BenchmarkCiPolicy, String> {
    let normalized = raw.trim().to_ascii_lowercase();

    match normalized.as_str() {
        "disabled" | "off" | "none" | "0" => Ok(BenchmarkCiPolicy::Disabled),
        "scheduled-baseline" | "scheduled_baseline" | "scheduled" | "weekly" | "nightly" => {
            Ok(BenchmarkCiPolicy::ScheduledBaseline)
        }
        "always-baseline" | "always_baseline" | "always" | "all" | "1" => {
            Ok(BenchmarkCiPolicy::AlwaysBaseline)
        }
        _ => Err("expected one of: disabled, scheduled-baseline, always-baseline".to_string()),
    }
}

/// Parses a GitHub event name into a benchmark CI event category.
#[must_use]
pub fn parse_event_name(raw: &str) -> BenchmarkCiEvent {
    match raw.trim().to_ascii_lowercase().as_str() {
        "pull_request" | "pull_request_target" => BenchmarkCiEvent::PullRequest,
        "schedule" => BenchmarkCiEvent::Schedule,
        "workflow_dispatch" => BenchmarkCiEvent::WorkflowDispatch,
        _ => BenchmarkCiEvent::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::collections::HashMap;

    fn load_with_overrides(
        default_policy: BenchmarkCiPolicy,
        event_override: Option<&str>,
        policy_override: Option<&str>,
    ) -> BenchmarkRegressionProfile {
        let mut env_entries: HashMap<&'static str, String> = HashMap::new();
        if let Some(raw) = event_override {
            env_entries.insert(GITHUB_EVENT_NAME_ENV_KEY, raw.to_owned());
        }
        if let Some(raw) = policy_override {
            env_entries.insert(CHUTORO_BENCH_CI_POLICY_ENV_KEY, raw.to_owned());
        }

        BenchmarkRegressionProfile::load_with_lookup(default_policy, |key| {
            env_entries.get(key).cloned()
        })
    }

    #[rstest]
    #[case("disabled", BenchmarkCiPolicy::Disabled)]
    #[case("OFF", BenchmarkCiPolicy::Disabled)]
    #[case("0", BenchmarkCiPolicy::Disabled)]
    #[case(" 0 ", BenchmarkCiPolicy::Disabled)]
    #[case("scheduled-baseline", BenchmarkCiPolicy::ScheduledBaseline)]
    #[case("scheduled_baseline", BenchmarkCiPolicy::ScheduledBaseline)]
    #[case("weekly", BenchmarkCiPolicy::ScheduledBaseline)]
    #[case("always-baseline", BenchmarkCiPolicy::AlwaysBaseline)]
    #[case("always_baseline", BenchmarkCiPolicy::AlwaysBaseline)]
    #[case("ALL", BenchmarkCiPolicy::AlwaysBaseline)]
    #[case("1", BenchmarkCiPolicy::AlwaysBaseline)]
    #[case(" 1 ", BenchmarkCiPolicy::AlwaysBaseline)]
    fn parse_policy_accepts_valid_values(#[case] raw: &str, #[case] expected: BenchmarkCiPolicy) {
        let parsed = parse_policy(raw).expect("valid policy should parse");
        assert_eq!(parsed, expected);
    }

    #[rstest]
    #[case("")]
    #[case(" ")]
    #[case("pr-only")]
    #[case("maybe")]
    fn parse_policy_rejects_invalid_values(#[case] raw: &str) {
        let err = parse_policy(raw).expect_err("invalid policy should fail");
        assert!(err.contains("expected one of"));
    }

    #[rstest]
    #[case("pull_request", BenchmarkCiEvent::PullRequest)]
    #[case("pull_request_target", BenchmarkCiEvent::PullRequest)]
    #[case("schedule", BenchmarkCiEvent::Schedule)]
    #[case("workflow_dispatch", BenchmarkCiEvent::WorkflowDispatch)]
    #[case("push", BenchmarkCiEvent::Other)]
    #[case("", BenchmarkCiEvent::Other)]
    fn parse_event_name_maps_expected_values(
        #[case] raw: &str,
        #[case] expected: BenchmarkCiEvent,
    ) {
        assert_eq!(parse_event_name(raw), expected);
    }

    #[rstest]
    #[case(
        BenchmarkCiEvent::PullRequest,
        BenchmarkCiPolicy::Disabled,
        BenchmarkRegressionMode::Disabled
    )]
    #[case(
        BenchmarkCiEvent::Schedule,
        BenchmarkCiPolicy::Disabled,
        BenchmarkRegressionMode::Disabled
    )]
    #[case(
        BenchmarkCiEvent::PullRequest,
        BenchmarkCiPolicy::ScheduledBaseline,
        BenchmarkRegressionMode::DiscoveryOnly
    )]
    #[case(
        BenchmarkCiEvent::WorkflowDispatch,
        BenchmarkCiPolicy::ScheduledBaseline,
        BenchmarkRegressionMode::BaselineCompare
    )]
    #[case(
        BenchmarkCiEvent::Schedule,
        BenchmarkCiPolicy::ScheduledBaseline,
        BenchmarkRegressionMode::BaselineCompare
    )]
    #[case(
        BenchmarkCiEvent::Other,
        BenchmarkCiPolicy::ScheduledBaseline,
        BenchmarkRegressionMode::DiscoveryOnly
    )]
    #[case(
        BenchmarkCiEvent::PullRequest,
        BenchmarkCiPolicy::AlwaysBaseline,
        BenchmarkRegressionMode::BaselineCompare
    )]
    fn resolve_regression_mode_maps_expected_values(
        #[case] event: BenchmarkCiEvent,
        #[case] policy: BenchmarkCiPolicy,
        #[case] expected: BenchmarkRegressionMode,
    ) {
        assert_eq!(resolve_regression_mode(event, policy), expected);
    }

    #[test]
    fn load_uses_defaults_when_no_overrides_exist() {
        let profile = load_with_overrides(BenchmarkCiPolicy::ScheduledBaseline, None, None);
        assert_eq!(profile.policy(), BenchmarkCiPolicy::ScheduledBaseline);
        assert_eq!(profile.event(), BenchmarkCiEvent::Other);
        assert_eq!(profile.mode(), BenchmarkRegressionMode::DiscoveryOnly);
    }

    #[test]
    fn load_uses_default_policy_when_override_is_invalid() {
        let profile = load_with_overrides(
            BenchmarkCiPolicy::AlwaysBaseline,
            Some("pull_request"),
            Some("invalid-policy"),
        );

        assert_eq!(profile.policy(), BenchmarkCiPolicy::AlwaysBaseline);
        assert_eq!(profile.mode(), BenchmarkRegressionMode::BaselineCompare);
    }

    #[rstest]
    #[case(
        Some("pull_request"),
        Some("scheduled-baseline"),
        BenchmarkRegressionMode::DiscoveryOnly
    )]
    #[case(
        Some("schedule"),
        Some("scheduled-baseline"),
        BenchmarkRegressionMode::BaselineCompare
    )]
    #[case(
        Some("workflow_dispatch"),
        Some("always-baseline"),
        BenchmarkRegressionMode::BaselineCompare
    )]
    #[case(Some("push"), Some("disabled"), BenchmarkRegressionMode::Disabled)]
    fn load_resolves_mode_from_event_and_policy(
        #[case] event_override: Option<&str>,
        #[case] policy_override: Option<&str>,
        #[case] expected_mode: BenchmarkRegressionMode,
    ) {
        let profile = load_with_overrides(
            BenchmarkCiPolicy::ScheduledBaseline,
            event_override,
            policy_override,
        );

        assert_eq!(profile.mode(), expected_mode);
    }
}
