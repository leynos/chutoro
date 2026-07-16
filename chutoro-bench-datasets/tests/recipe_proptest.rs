//! Property tests for source-order preservation.

use bytes::Bytes;
use chutoro_bench_datasets::{
    RecipeContext, SourceUrl, run_recipe,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe},
};
use chutoro_test_support::ci::property_test_profile::{PROPTEST_RNG_SEED, ProptestRunProfile};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, RngSeed};

fn suite_proptest_config(default_cases: u32) -> ProptestConfig {
    let profile = ProptestRunProfile::load(default_cases, false);
    ProptestConfig {
        cases: profile.cases(),
        fork: profile.fork(),
        rng_seed: RngSeed::Fixed(PROPTEST_RNG_SEED),
        ..ProptestConfig::default()
    }
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
