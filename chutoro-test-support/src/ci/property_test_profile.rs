//! Property-test run profile parsing for CI and local overrides.
//!
//! This module centralizes environment-driven proptest tuning so multiple
//! suites can share one policy surface.

use mockable::{DefaultEnv, Env};

/// Environment variable controlling proptest case counts.
pub const PROPTEST_CASES_ENV_KEY: &str = "PROPTEST_CASES";
/// Legacy environment variable controlling proptest case counts.
pub const PROGTEST_CASES_ENV_KEY: &str = "PROGTEST_CASES";
/// Environment variable controlling proptest process forking.
pub const CHUTORO_PBT_FORK_ENV_KEY: &str = "CHUTORO_PBT_FORK";

/// Fixed RNG seed applied to every property suite so coverage is deterministic.
///
/// By default proptest seeds its runner from operating-system entropy, so each
/// run explores a different pseudo-random set of cases and touches a different
/// set of lines. That makes whole-workspace line coverage wobble run-to-run for
/// identical code and false-trips the coverage ratchet. Pinning the seed keeps
/// proptest's exploratory value (it still walks a fixed pseudo-random set) while
/// making the covered lines reproducible. Suites that generate their own RNG
/// seeds through proptest strategies (for example HNSW build seeds) inherit this
/// determinism, because those draws are taken from the seeded runner.
///
/// Apply it by setting `rng_seed: proptest::test_runner::RngSeed::Fixed(..)` on
/// the `proptest::test_runner::Config` used by a suite.
pub const PROPTEST_RNG_SEED: u64 = 0x600D_5EED_C047_0207;

/// Global rejects proptest allows per case asked for.
///
/// proptest's own defaults are 256 cases against 1024 global rejects, so four
/// is the ratio the library itself implies. The budget is a total for the run,
/// not a per-case allowance, which is why it has to scale with the case count.
const GLOBAL_REJECTS_PER_CASE: u32 = 4;

/// proptest's own `max_global_rejects`, used as a floor.
///
/// Keeping it means a run at or below proptest's default case count is
/// unchanged, so the 250-case pull-request lane behaves exactly as before.
///
/// Public because a test asserting that a config derives its budget has to be
/// able to say that its fixture is deep enough to tell the derivation from
/// this default. At or below the floor the two agree, and such a test passes
/// whether the wiring is there or not.
pub const DEFAULT_MAX_GLOBAL_REJECTS: u32 = 1024;

/// Global reject budget for a run of `cases` cases.
///
/// A suite whose strategies filter their inputs spends rejects across the
/// whole run while proptest's budget stays flat at 1024 however many cases are
/// asked for. At the pull-request lane's 250 cases that is never approached.
/// At the weekly lane's 25,000 it is certain: the HNSW search and mutation
/// suites reached 736 and 7,326 successes before aborting on it, the first
/// time they ever ran (#260).
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::property_test_profile::max_global_rejects_for;
///
/// // At or below proptest's default case count, nothing changes.
/// assert_eq!(max_global_rejects_for(250), 1024);
/// // A deep run gets a budget in proportion to what it asks for.
/// assert_eq!(max_global_rejects_for(25_000), 100_000);
/// ```
#[must_use]
pub const fn max_global_rejects_for(cases: u32) -> u32 {
    let scaled = cases.saturating_mul(GLOBAL_REJECTS_PER_CASE);
    if scaled < DEFAULT_MAX_GLOBAL_REJECTS {
        DEFAULT_MAX_GLOBAL_REJECTS
    } else {
        scaled
    }
}

/// Runtime profile for property-test execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProptestRunProfile {
    /// Number of cases each property test should execute.
    cases: u32,
    /// Whether proptest should isolate cases in subprocesses.
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
        Self::load_with_env(default_cases, default_fork, &DefaultEnv)
    }

    /// Number of cases to run per property.
    #[must_use]
    pub const fn cases(&self) -> u32 {
        self.cases
    }

    /// Whether to run proptest cases in forked subprocesses.
    #[must_use]
    pub const fn fork(&self) -> bool {
        self.fork
    }

    /// Global reject budget matching this profile's case count.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::property_test_profile::ProptestRunProfile;
    ///
    /// let profile = ProptestRunProfile::load(64, false);
    /// assert!(profile.max_global_rejects() >= 1024);
    /// ```
    #[must_use]
    pub const fn max_global_rejects(&self) -> u32 {
        max_global_rejects_for(self.cases)
    }

    /// Load through an injected reader so tests avoid process-global state.
    fn load_with_env(default_cases: u32, default_fork: bool, env: &dyn Env) -> Self {
        let cases = read_cases_or_default(default_cases, env);
        let fork = read_env_or_default(CHUTORO_PBT_FORK_ENV_KEY, default_fork, parse_bool, env);
        Self { cases, fork }
    }
}

/// Read the current or legacy cases override, otherwise return `default`.
fn read_cases_or_default(default: u32, env: &dyn Env) -> u32 {
    read_env(PROPTEST_CASES_ENV_KEY, parse_cases, env).map_or_else(
        || read_env_or_default(PROGTEST_CASES_ENV_KEY, default, parse_cases, env),
        |cases| cases.unwrap_or(default),
    )
}

/// Parse an optional environment value, falling back to `default` on failure.
fn read_env_or_default<T, F>(key: &'static str, default: T, parser: F, env: &dyn Env) -> T
where
    T: Copy,
    F: Fn(&str) -> Result<T, String>,
{
    read_env(key, parser, env).map_or(default, |value| value.unwrap_or(default))
}

/// Read and parse an optional environment value, logging invalid overrides.
fn read_env<T, F>(key: &'static str, parser: F, env: &dyn Env) -> Option<Result<T, String>>
where
    F: Fn(&str) -> Result<T, String>,
{
    env.string(key).map(|raw| {
        parser(&raw).map_err(|reason| {
            tracing::warn!(
                env = key,
                raw = %raw,
                reason = %reason,
                "invalid property-test profile override; using default",
            );
            reason
        })
    })
}

/// Parse a non-zero proptest case count from an environment value.
fn parse_cases(raw: &str) -> Result<u32, String> {
    let parsed = raw
        .trim()
        .parse::<u32>()
        .map_err(|error| format!("parse error: {error}"))?;
    if parsed == 0 {
        return Err("cases must be > 0".to_owned());
    }
    Ok(parsed)
}

/// Parse an accepted boolean spelling from an environment value.
fn parse_bool(raw: &str) -> Result<bool, String> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err("expected one of: true/false/1/0/yes/no/on/off".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for property-test profile selection.

    use super::*;
    use mockable::MockEnv;
    use rstest::rstest;
    use std::collections::HashMap;

    #[derive(Clone, Copy, Default)]
    struct ProfileOverrides<'a> {
        cases: Option<&'a str>,
        legacy_cases: Option<&'a str>,
        fork: Option<&'a str>,
    }

    fn load_with_overrides(
        default_cases: u32,
        default_fork: bool,
        overrides: ProfileOverrides<'_>,
    ) -> ProptestRunProfile {
        let mut env_entries: HashMap<&'static str, String> = HashMap::new();
        if let Some(raw) = overrides.cases {
            env_entries.insert(PROPTEST_CASES_ENV_KEY, raw.to_owned());
        }
        if let Some(raw) = overrides.legacy_cases {
            env_entries.insert(PROGTEST_CASES_ENV_KEY, raw.to_owned());
        }
        if let Some(raw) = overrides.fork {
            env_entries.insert(CHUTORO_PBT_FORK_ENV_KEY, raw.to_owned());
        }

        let mut env = MockEnv::new();
        env.expect_string()
            .returning(move |key| env_entries.get(key).cloned());
        ProptestRunProfile::load_with_env(default_cases, default_fork, &env)
    }

    #[test]
    fn load_defaults_when_no_overrides_exist() {
        let profile = load_with_overrides(64, false, ProfileOverrides::default());
        assert_eq!(profile.cases(), 64);
        assert!(!profile.fork());
    }

    #[rstest]
    #[case("1", 1)]
    #[case("250", 250)]
    #[case("25000", 25_000)]
    fn load_accepts_valid_case_overrides(#[case] raw: &str, #[case] expected: u32) {
        let profile = load_with_overrides(
            64,
            false,
            ProfileOverrides {
                cases: Some(raw),
                ..ProfileOverrides::default()
            },
        );
        assert_eq!(profile.cases(), expected);
    }

    #[test]
    fn load_falls_back_to_legacy_case_override() {
        let profile = load_with_overrides(
            64,
            false,
            ProfileOverrides {
                legacy_cases: Some("512"),
                ..ProfileOverrides::default()
            },
        );
        assert_eq!(profile.cases(), 512);
    }

    #[test]
    fn load_prefers_standard_case_override_over_legacy() {
        let profile = load_with_overrides(
            64,
            false,
            ProfileOverrides {
                cases: Some("128"),
                legacy_cases: Some("512"),
                ..ProfileOverrides::default()
            },
        );
        assert_eq!(profile.cases(), 128);
    }

    #[test]
    fn load_does_not_fall_back_to_legacy_when_standard_override_is_invalid() {
        let profile = load_with_overrides(
            64,
            false,
            ProfileOverrides {
                cases: Some("0"),
                legacy_cases: Some("512"),
                ..ProfileOverrides::default()
            },
        );
        assert_eq!(profile.cases(), 64);
    }

    #[rstest]
    #[case("0")]
    #[case("-1")]
    #[case("abc")]
    fn load_rejects_invalid_case_overrides(#[case] raw: &str) {
        let profile = load_with_overrides(
            64,
            false,
            ProfileOverrides {
                cases: Some(raw),
                ..ProfileOverrides::default()
            },
        );
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
        let profile = load_with_overrides(
            64,
            false,
            ProfileOverrides {
                fork: Some(raw),
                ..ProfileOverrides::default()
            },
        );
        assert_eq!(profile.fork(), expected);
    }

    #[rstest]
    #[case("")]
    #[case("maybe")]
    #[case("2")]
    fn load_rejects_invalid_fork_overrides(#[case] raw: &str) {
        let profile = load_with_overrides(
            64,
            true,
            ProfileOverrides {
                fork: Some(raw),
                ..ProfileOverrides::default()
            },
        );
        assert!(profile.fork());
    }
}
