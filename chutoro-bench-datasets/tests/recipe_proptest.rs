//! Property tests for source-order preservation.

use bytes::Bytes;
use chutoro_bench_datasets::{
    RecipeContext, SourceUrl, run_recipe,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe},
};
use chutoro_test_support::ci::property_test_profile::{
    DEFAULT_MAX_GLOBAL_REJECTS, PROPTEST_RNG_SEED, ProptestRunProfile, max_global_rejects_for,
};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, RngSeed};

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

/// The recipe suite config sizes its reject budget from its case count.
///
/// proptest's flat 1024 default is a total for the run, not a per-case
/// allowance, so a deep run exhausts it however many cases it was asked for
/// (#260).
#[test]
fn the_recipe_config_scales_its_reject_budget() {
    let config = suite_proptest_config(25_000);

    assert!(
        max_global_rejects_for(config.cases) > DEFAULT_MAX_GLOBAL_REJECTS,
        "the fixture must ask for a run deep enough that the floor is not \
         the answer, or this test passes whether the budget is derived or \
         left on proptest's default"
    );

    assert_eq!(
        config.max_global_rejects,
        max_global_rejects_for(config.cases),
        "a deep run left on proptest's flat default aborts before it finishes"
    );
}
