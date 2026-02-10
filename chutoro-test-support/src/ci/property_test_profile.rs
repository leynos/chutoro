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
        let cases = read_env_or_default(PROGTEST_CASES_ENV_KEY, default_cases, parse_cases);
        let fork = read_env_or_default(CHUTORO_PBT_FORK_ENV_KEY, default_fork, parse_bool);
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

fn read_env_or_default<T, F>(key: &'static str, default: T, parser: F) -> T
where
    T: Copy,
    F: Fn(&str) -> Result<T, String>,
{
    match env::var(key) {
        Ok(raw) => match parser(&raw) {
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
        Err(_) => default,
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
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = env::var(key).ok();
            // SAFETY: tests serialize access with ENV_LOCK.
            unsafe { env::set_var(key, value) };
            Self { key, original }
        }

        fn unset(key: &'static str) -> Self {
            let original = env::var(key).ok();
            // SAFETY: tests serialize access with ENV_LOCK.
            unsafe { env::remove_var(key) };
            Self { key, original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.original {
                // SAFETY: tests serialize access with ENV_LOCK.
                unsafe { env::set_var(self.key, value) };
            } else {
                // SAFETY: tests serialize access with ENV_LOCK.
                unsafe { env::remove_var(self.key) };
            }
        }
    }

    #[test]
    fn load_defaults_when_no_overrides_exist() {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _cases = EnvGuard::unset(PROGTEST_CASES_ENV_KEY);
        let _fork = EnvGuard::unset(CHUTORO_PBT_FORK_ENV_KEY);

        let profile = ProptestRunProfile::load(64, false);
        assert_eq!(profile.cases(), 64);
        assert!(!profile.fork());
    }

    #[rstest]
    #[case("1", 1)]
    #[case("250", 250)]
    #[case("25000", 25_000)]
    fn load_accepts_valid_case_overrides(#[case] raw: &str, #[case] expected: u32) {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _cases = EnvGuard::set(PROGTEST_CASES_ENV_KEY, raw);
        let _fork = EnvGuard::unset(CHUTORO_PBT_FORK_ENV_KEY);

        let profile = ProptestRunProfile::load(64, false);
        assert_eq!(profile.cases(), expected);
    }

    #[rstest]
    #[case("0")]
    #[case("-1")]
    #[case("abc")]
    fn load_rejects_invalid_case_overrides(#[case] raw: &str) {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _cases = EnvGuard::set(PROGTEST_CASES_ENV_KEY, raw);
        let _fork = EnvGuard::unset(CHUTORO_PBT_FORK_ENV_KEY);

        let profile = ProptestRunProfile::load(64, false);
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
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _cases = EnvGuard::unset(PROGTEST_CASES_ENV_KEY);
        let _fork = EnvGuard::set(CHUTORO_PBT_FORK_ENV_KEY, raw);

        let profile = ProptestRunProfile::load(64, false);
        assert_eq!(profile.fork(), expected);
    }

    #[rstest]
    #[case("")]
    #[case("maybe")]
    #[case("2")]
    fn load_rejects_invalid_fork_overrides(#[case] raw: &str) {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _cases = EnvGuard::unset(PROGTEST_CASES_ENV_KEY);
        let _fork = EnvGuard::set(CHUTORO_PBT_FORK_ENV_KEY, raw);

        let profile = ProptestRunProfile::load(64, true);
        assert!(profile.fork());
    }
}
