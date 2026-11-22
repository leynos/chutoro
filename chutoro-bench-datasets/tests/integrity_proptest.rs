//! Property tests for checksum parsing and canonical display.

use chutoro_bench_datasets::{Checksum, DigestAlgorithm};
use chutoro_test_support::ci::property_test_profile::{PROPTEST_RNG_SEED, ProptestRunProfile};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, RngSeed};

/// Build the suite proptest configuration shared with the recipe properties.
fn suite_proptest_config(default_cases: u32) -> ProptestConfig {
    let profile = ProptestRunProfile::load(default_cases, false);
    ProptestConfig {
        cases: profile.cases(),
        fork: profile.fork(),
        // Rejects are a budget for the whole run, so a deep run needs one in
        // proportion to the cases it asks for. See #260.
        max_global_rejects: profile.max_global_rejects(),
        rng_seed: RngSeed::Fixed(PROPTEST_RNG_SEED),
        ..ProptestConfig::default()
    }
}

/// Report whether `value` matches `^sha256:[0-9A-Fa-f]{64}$`.
fn is_valid_checksum_text(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|payload| {
        payload.len() == 64 && payload.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

proptest! {
    #![proptest_config(suite_proptest_config(256))]

    #[test]
    fn checksum_round_trips_through_display(bytes in any::<[u8; 32]>()) {
        let checksum = Checksum::new(DigestAlgorithm::Sha256, bytes);
        let text = checksum.to_string();

        prop_assert!(text.starts_with("sha256:"));
        let parsed = Checksum::parse(&text)?;
        prop_assert_eq!(&parsed, &checksum);
        prop_assert_eq!(parsed.as_bytes(), &bytes);
    }

    #[test]
    fn checksum_parse_rejects_text_outside_the_grammar(value in any::<String>()) {
        prop_assume!(!is_valid_checksum_text(&value));
        prop_assert!(Checksum::parse(&value).is_err());
    }

    #[test]
    fn checksum_parse_rejects_non_hex_digest_bodies(payload in "[ -~]{64}") {
        prop_assume!(!payload.bytes().all(|byte| byte.is_ascii_hexdigit()));
        let text = format!("sha256:{payload}");
        prop_assert!(Checksum::parse(&text).is_err());
    }
}
