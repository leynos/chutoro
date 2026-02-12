//! Configuration and parsing for the HNSW search-correctness property.
//!
//! Reads environment overrides for the minimum recall threshold and maximum
//! fixture length used by the property-based search tests.

use std::env;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum RecallThresholdError {
    ParseFloat,
    OutOfRange { value: f32 },
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SearchPropertyConfig {
    min_recall: f32,
    max_fixture_len: usize,
    min_max_connections: usize,
}

#[derive(Clone, Copy, Debug)]
struct EnvKey(&'static str);

impl EnvKey {
    const fn as_str(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
struct RawConfigValue<'a>(&'a str);

impl<'a> RawConfigValue<'a> {
    fn trimmed(self) -> &'a str {
        self.0.trim()
    }
}

impl SearchPropertyConfig {
    const ENV_KEY: EnvKey = EnvKey("CHUTORO_HNSW_PBT_MIN_RECALL");
    const MAX_FIXTURE_LEN_ENV_KEY: EnvKey = EnvKey("CHUTORO_HNSW_PBT_MAX_FIXTURE_LEN");
    const MIN_MAX_CONNECTIONS_ENV_KEY: EnvKey = EnvKey("CHUTORO_HNSW_PBT_MIN_MAX_CONNECTIONS");
    pub(super) const DEFAULT_MIN_RECALL: f32 = 0.50;
    pub(super) const DEFAULT_MAX_FIXTURE_LEN: usize = 32;
    pub(super) const DEFAULT_MIN_MAX_CONNECTIONS: usize = 12;

    pub(super) fn load() -> Self {
        let min_recall = Self::read_env_or_default(
            Self::ENV_KEY,
            Self::DEFAULT_MIN_RECALL,
            Self::parse_min_recall,
        );
        let max_fixture_len = Self::read_env_or_default(
            Self::MAX_FIXTURE_LEN_ENV_KEY,
            Self::DEFAULT_MAX_FIXTURE_LEN,
            Self::parse_max_fixture_len,
        );
        let min_max_connections = Self::read_env_or_default(
            Self::MIN_MAX_CONNECTIONS_ENV_KEY,
            Self::DEFAULT_MIN_MAX_CONNECTIONS,
            Self::parse_min_max_connections,
        );

        Self {
            min_recall,
            max_fixture_len,
            min_max_connections,
        }
    }

    pub(super) fn min_recall(&self) -> f32 {
        self.min_recall
    }

    pub(super) fn max_fixture_len(&self) -> usize {
        self.max_fixture_len
    }

    pub(super) fn min_max_connections(&self) -> usize {
        self.min_max_connections
    }

    fn read_env_or_default<T, F>(key: EnvKey, default: T, parser: F) -> T
    where
        T: Copy,
        F: for<'a> Fn(RawConfigValue<'a>) -> Result<T, String>,
    {
        match env::var(key.as_str()) {
            Ok(raw) => match parser(RawConfigValue(raw.as_str())) {
                Ok(value) => value,
                Err(reason) => {
                    tracing::warn!(
                        env = key.as_str(),
                        raw = %raw,
                        reason = %reason,
                        "invalid config override, falling back to default",
                    );
                    default
                }
            },
            Err(_) => default,
        }
    }

    fn parse_min_recall(raw: RawConfigValue<'_>) -> Result<f32, String> {
        parse_recall_threshold(raw).map_err(|err| format!("{err:?}"))
    }

    fn parse_max_fixture_len(raw: RawConfigValue<'_>) -> Result<usize, String> {
        Self::parse_usize_with_min(raw, 2)
    }

    fn parse_min_max_connections(raw: RawConfigValue<'_>) -> Result<usize, String> {
        Self::parse_usize_with_min(raw, 2)
    }

    fn parse_usize_with_min(raw: RawConfigValue<'_>, min: usize) -> Result<usize, String> {
        let trimmed = raw.trimmed();
        let parsed = trimmed
            .parse::<usize>()
            .map_err(|err| format!("parse error: {err}"))?;
        if parsed < min {
            return Err(format!("value must be >= {min}"));
        }
        Ok(parsed)
    }
}

fn parse_recall_threshold(raw: RawConfigValue<'_>) -> Result<f32, RecallThresholdError> {
    let trimmed = raw.trimmed();
    let parsed = trimmed
        .parse::<f32>()
        .map_err(|_| RecallThresholdError::ParseFloat)?;
    if parsed <= 0.0 || parsed > 1.0 {
        return Err(RecallThresholdError::OutOfRange { value: parsed });
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("0.5", 0.5)]
    #[case("1.0", 1.0)]
    #[case("0.01", 0.01)]
    fn parse_recall_threshold_accepts_valid_values(#[case] input: &str, #[case] expected: f32) {
        let parsed = parse_recall_threshold(RawConfigValue(input)).expect("value should parse");
        assert!(
            (parsed - expected).abs() < f32::EPSILON,
            "parsed {parsed} vs {expected}"
        );
    }

    #[rstest]
    #[case("0.0", RecallThresholdError::OutOfRange { value: 0.0 })]
    #[case("-0.1", RecallThresholdError::OutOfRange { value: -0.1 })]
    #[case("1.0001", RecallThresholdError::OutOfRange { value: 1.0001 })]
    #[case("abc", RecallThresholdError::ParseFloat)]
    fn parse_recall_threshold_rejects_invalid_values(
        #[case] input: &str,
        #[case] expected: RecallThresholdError,
    ) {
        let err = parse_recall_threshold(RawConfigValue(input)).expect_err("value should fail");
        assert_eq!(err, expected);
    }

    #[rstest]
    #[case("2", 2)]
    #[case("8", 8)]
    #[case("16", 16)]
    fn parse_min_max_connections_accepts_valid_values(
        #[case] input: &str,
        #[case] expected: usize,
    ) {
        let parsed = SearchPropertyConfig::parse_min_max_connections(RawConfigValue(input))
            .expect("value should parse");
        assert_eq!(parsed, expected);
    }

    #[rstest]
    #[case("0")]
    #[case("1")]
    #[case("-1")]
    #[case("abc")]
    fn parse_min_max_connections_rejects_invalid_values(#[case] input: &str) {
        let err = SearchPropertyConfig::parse_min_max_connections(RawConfigValue(input))
            .expect_err("value should fail");
        assert!(!err.is_empty(), "error message should be non-empty");
    }
}
