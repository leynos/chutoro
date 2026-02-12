//! Property-test run profile parsing for CI and local overrides.
//!
//! This module centralizes environment-driven proptest tuning so multiple
//! suites can share one policy surface.

use std::env;

/// Environment variable controlling proptest case counts.
pub const PROGTEST_CASES_ENV_KEY: &str = "PROGTEST_CASES";
/// Environment variable controlling proptest process forking.
pub const CHUTORO_PBT_FORK_ENV_KEY: &str = "CHUTORO_PBT_FORK";

/// Runtime profile for property-test execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProptestRunProfile {
    cases: u32,
    fork: bool,
}

impl ProptestRunProfile {
    /// Load a profile from environment variables with provided defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::property_test_profile::ProptestRunProfile;
    ///
    /// let profile = ProptestRunProfile::load(64, false);
    /// assert!(profile.cases() > 0);
    /// ```
    #[must_use]
    pub fn load(default_cases: u32, default_fork: bool) -> Self {
        Self::load_with_lookup(default_cases, default_fork, |key| env::var(key).ok())
    }

    fn load_with_lookup<F>(default_cases: u32, default_fork: bool, lookup: F) -> Self
    where
        F: Fn(&'static str) -> Option<String>,
    {
        let cases =
            read_env_or_default(PROGTEST_CASES_ENV_KEY, default_cases, parse_cases, &lookup);
        let fork = read_env_or_default(CHUTORO_PBT_FORK_ENV_KEY, default_fork, parse_bool, &lookup);
        Self { cases, fork }
    }

    /// Number of cases to run per property.
    #[must_use]
    pub fn cases(&self) -> u32 {
        self.cases
    }

    /// Whether to run proptest cases in forked subprocesses.
    #[must_use]
    pub fn fork(&self) -> bool {
        self.fork
    }
}

fn read_env_or_default<T, F, L>(key: &'static str, default: T, parser: F, lookup: &L) -> T
where
    T: Copy,
    F: Fn(&str) -> Result<T, String>,
    L: Fn(&'static str) -> Option<String>,
{
    match lookup(key) {
        Some(raw) => match parser(&raw) {
            Ok(value) => value,
            Err(reason) => {
                tracing::warn!(
                    env = key,
                    raw = %raw,
                    reason = %reason,
                    "invalid property-test profile override; using default",
                );
                default
            }
        },
        None => default,
    }
}

fn parse_cases(raw: &str) -> Result<u32, String> {
    let parsed = raw
        .trim()
        .parse::<u32>()
        .map_err(|error| format!("parse error: {error}"))?;
    if parsed == 0 {
        return Err("cases must be > 0".to_string());
    }
    Ok(parsed)
}

fn parse_bool(raw: &str) -> Result<bool, String> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err("expected one of: true/false/1/0/yes/no/on/off".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::collections::HashMap;

    fn load_with_overrides(
        default_cases: u32,
        default_fork: bool,
        cases_override: Option<&str>,
        fork_override: Option<&str>,
    ) -> ProptestRunProfile {
        let mut env_entries: HashMap<&'static str, String> = HashMap::new();
        if let Some(raw) = cases_override {
            env_entries.insert(PROGTEST_CASES_ENV_KEY, raw.to_owned());
        }
        if let Some(raw) = fork_override {
            env_entries.insert(CHUTORO_PBT_FORK_ENV_KEY, raw.to_owned());
        }

        ProptestRunProfile::load_with_lookup(default_cases, default_fork, |key| {
            env_entries.get(key).cloned()
        })
    }

    #[test]
    fn load_defaults_when_no_overrides_exist() {
        let profile = load_with_overrides(64, false, None, None);
        assert_eq!(profile.cases(), 64);
        assert!(!profile.fork());
    }

    #[rstest]
    #[case("1", 1)]
    #[case("250", 250)]
    #[case("25000", 25_000)]
    fn load_accepts_valid_case_overrides(#[case] raw: &str, #[case] expected: u32) {
        let profile = load_with_overrides(64, false, Some(raw), None);
        assert_eq!(profile.cases(), expected);
    }

    #[rstest]
    #[case("0")]
    #[case("-1")]
    #[case("abc")]
    fn load_rejects_invalid_case_overrides(#[case] raw: &str) {
        let profile = load_with_overrides(64, false, Some(raw), None);
        assert_eq!(profile.cases(), 64);
    }

    #[rstest]
    #[case("true", true)]
    #[case("TRUE", true)]
    #[case("1", true)]
    #[case("yes", true)]
    #[case("on", true)]
    #[case("false", false)]
    #[case("FALSE", false)]
    #[case("0", false)]
    #[case("no", false)]
    #[case("off", false)]
    fn load_accepts_valid_fork_overrides(#[case] raw: &str, #[case] expected: bool) {
        let profile = load_with_overrides(64, false, None, Some(raw));
        assert_eq!(profile.fork(), expected);
    }

    #[rstest]
    #[case("")]
    #[case("maybe")]
    #[case("2")]
    fn load_rejects_invalid_fork_overrides(#[case] raw: &str) {
        let profile = load_with_overrides(64, true, None, Some(raw));
        assert!(profile.fork());
    }
}
