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
}

impl SearchPropertyConfig {
    pub(super) const ENV_KEY: &'static str = "CHUTORO_HNSW_PBT_MIN_RECALL";
    pub(super) const MAX_FIXTURE_LEN_ENV_KEY: &'static str = "CHUTORO_HNSW_PBT_MAX_FIXTURE_LEN";
    pub(super) const DEFAULT_MIN_RECALL: f32 = 0.50;
    pub(super) const DEFAULT_MAX_FIXTURE_LEN: usize = 32;

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

        Self {
            min_recall,
            max_fixture_len,
        }
    }

    pub(super) fn min_recall(&self) -> f32 {
        self.min_recall
    }

    pub(super) fn max_fixture_len(&self) -> usize {
        self.max_fixture_len
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
                        "invalid config override, falling back to default",
                    );
                    default
                }
            },
            Err(_) => default,
        }
    }

    fn parse_min_recall(raw: &str) -> Result<f32, String> {
        parse_recall_threshold(raw).map_err(|err| format!("{err:?}"))
    }

    fn parse_max_fixture_len(raw: &str) -> Result<usize, String> {
        let trimmed = raw.trim();
        let parsed = trimmed
            .parse::<usize>()
            .map_err(|err| format!("parse error: {err}"))?;
        if parsed < 2 {
            return Err("value must be >= 2".to_string());
        }
        Ok(parsed)
    }
}

pub(super) fn parse_recall_threshold(raw: &str) -> Result<f32, RecallThresholdError> {
    let trimmed = raw.trim();
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
        let parsed = parse_recall_threshold(input).expect("value should parse");
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
        let err = parse_recall_threshold(input).expect_err("value should fail");
        assert_eq!(err, expected);
    }
}
