//! Property tests for source-order preservation.

use bytes::Bytes;
use chutoro_bench_datasets::{
    RecipeContext, SourceUrl, run_recipe,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe},
};
use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;

const PROPTEST_CASES_ENV_KEY: &str = "PROPTEST_CASES";
// Legacy fallback for the previous "PROGTEST" typo used by workspace CI.
const LEGACY_PROGTEST_CASES_ENV_KEY: &str = "PROGTEST_CASES";
const CHUTORO_PBT_FORK_ENV_KEY: &str = "CHUTORO_PBT_FORK";

fn suite_proptest_config(default_cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases: read_cases(default_cases),
        fork: read_bool(CHUTORO_PBT_FORK_ENV_KEY, false),
        ..ProptestConfig::default()
    }
}

fn read_cases(default_cases: u32) -> u32 {
    std::env::var(PROPTEST_CASES_ENV_KEY)
        .or_else(|_error| std::env::var(LEGACY_PROGTEST_CASES_ENV_KEY))
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        .filter(|cases| *cases > 0)
        .unwrap_or(default_cases)
}

fn read_bool(key: &str, default_value: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default_value)
}

proptest! {
    #![proptest_config(suite_proptest_config(256))]

    #[test]
    fn stub_recipe_fetches_sources_in_declared_order(indices in prop::collection::vec(0u8..64, 1..32)) {
        let urls = indices
            .iter()
            .map(|index| SourceUrl::parse(&format!("https://example.test/source-{index}.bin")))
            .collect::<Result<Vec<_>, _>>()?;
        let entries = urls
            .iter()
            .cloned()
            .map(|url| (url, Bytes::from_static(b"x")))
            .collect::<Vec<_>>();
        let fetcher = InMemoryFetcher::new(entries);
        let storage = InMemoryStorage::default();
        let publisher = InMemoryPublisher::default();
        let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
        let recipe = StubRecipe::new("ordered", urls.clone());

        run_recipe(&recipe, &ctx)?;

        prop_assert_eq!(fetcher.requested_urls()?, urls);
    }
}
