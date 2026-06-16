//! Unit tests for the dataset recipe lifecycle driver.

use bytes::Bytes;
use chutoro_bench_datasets::{
    DatasetInfo, DatasetRecipe, ManifestDigest, ObjectKey, PartialState, PublishedArtefact,
    PublishedManifest, RecipeContext, RecipeError, RecipeId, RecipeVersion, SourceSpec, SourceUrl,
    run_recipe,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe},
};
use rstest::rstest;
use tracing_test::traced_test;

fn source_url() -> SourceUrl {
    SourceUrl::parse("https://example.test/data.bin")
        .unwrap_or_else(|error| panic!("test source URL should parse: {error}"))
}

#[traced_test]
#[test]
fn run_recipe_publishes_prepared_bytes() {
    let source = source_url();
    let fetcher = InMemoryFetcher::new([(source.clone(), Bytes::from_static(b"abc"))]);
    let storage = InMemoryStorage::default();
    let publish_sink = InMemoryPublisher::default();
    let ctx = RecipeContext::new(&fetcher, &storage, &publish_sink);
    let recipe = StubRecipe::new("stub", vec![source]);

    let artefact = run_recipe(&recipe, &ctx).expect("stub recipe should run");

    assert_eq!(artefact.manifest_uri().as_str(), "manifests/stub.json");
    assert!(logs_contain("executing dataset recipe phase"));
    let records = publish_sink
        .records()
        .expect("published records should be readable");
    assert_eq!(
        records.get(&ObjectKey::new("prepared/stub.bin")),
        Some(&Bytes::from_static(b"abc")),
    );
}

#[test]
fn fetch_size_limit_is_propagated() {
    let source = source_url();
    let fetcher = InMemoryFetcher::new([(source.clone(), Bytes::from_static(b"abcd"))]);
    let storage = InMemoryStorage::default();
    let publisher = InMemoryPublisher::default();
    let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
    let recipe = StubRecipe::new("stub", vec![source]).with_max_bytes(3);

    let Err(error) = run_recipe(&recipe, &ctx) else {
        panic!("oversized source should fail");
    };

    assert!(matches!(
        error,
        RecipeError::FetchSizeExceeded { limit_bytes: 3, .. }
    ));
}

#[rstest]
#[case::fetch(FailingPhase::Fetch, None)]
#[case::validate(FailingPhase::Validate, Some(chutoro_bench_datasets::Phase::Fetch))]
#[case::prepare(FailingPhase::Prepare, Some(chutoro_bench_datasets::Phase::Validate))]
#[case::publish(FailingPhase::Publish, Some(chutoro_bench_datasets::Phase::Prepare))]
fn failure_invokes_cleanup_with_partial_state(
    #[case] failing_phase: FailingPhase,
    #[case] expected_completed_phase: Option<chutoro_bench_datasets::Phase>,
) {
    let source = source_url();
    let fetcher = InMemoryFetcher::new([(source.clone(), Bytes::from_static(b"abc"))]);
    let storage = InMemoryStorage::default();
    let publisher = InMemoryPublisher::default();
    let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
    let recipe = FailingRecipe::new(source, failing_phase);

    let Err(error) = run_recipe(&recipe, &ctx) else {
        panic!("configured phase should fail");
    };

    assert!(error.to_string().contains(failing_phase.message()));
    assert_eq!(
        recipe.cleanup_state(),
        Some(PartialState::new(expected_completed_phase)),
    );
}

#[derive(Clone, Copy, Debug)]
enum FailingPhase {
    Fetch,
    Validate,
    Prepare,
    Publish,
}

impl FailingPhase {
    const fn message(self) -> &'static str {
        match self {
            Self::Fetch => "fetch failed",
            Self::Validate => "validate failed",
            Self::Prepare => "prepare failed",
            Self::Publish => "publish failed",
        }
    }

    const fn phase(self) -> chutoro_bench_datasets::Phase {
        match self {
            Self::Fetch => chutoro_bench_datasets::Phase::Fetch,
            Self::Validate => chutoro_bench_datasets::Phase::Validate,
            Self::Prepare => chutoro_bench_datasets::Phase::Prepare,
            Self::Publish => chutoro_bench_datasets::Phase::Publish,
        }
    }
}

#[derive(Debug)]
struct FailingRecipe {
    source: SourceUrl,
    sources: Vec<SourceSpec>,
    failing_phase: FailingPhase,
    cleanup_state: std::sync::Mutex<Option<PartialState>>,
}

impl FailingRecipe {
    fn new(source: SourceUrl, failing_phase: FailingPhase) -> Self {
        Self {
            sources: vec![SourceSpec::primary(source.clone())],
            source,
            failing_phase,
            cleanup_state: std::sync::Mutex::new(None),
        }
    }

    fn cleanup_state(&self) -> Option<PartialState> {
        match self.cleanup_state.lock() {
            Ok(state) => state.clone(),
            Err(error) => panic!("cleanup state lock should not be poisoned: {error}"),
        }
    }
}

impl DatasetRecipe for FailingRecipe {
    type Fetched = Bytes;
    type Validated = Bytes;
    type Prepared = Bytes;
    type Published = PublishedManifest;

    fn id(&self) -> RecipeId {
        RecipeId::new("failing")
    }

    fn version(&self) -> RecipeVersion {
        RecipeVersion::new(0, 1, 0)
    }

    fn info(&self) -> DatasetInfo {
        DatasetInfo::new(self.id(), self.version())
    }

    fn sources(&self) -> &[SourceSpec] {
        &self.sources
    }

    fn fetch(&self, ctx: &RecipeContext<'_>) -> Result<Self::Fetched, RecipeError> {
        if matches!(self.failing_phase, FailingPhase::Fetch) {
            Err(RecipeError::port(
                chutoro_bench_datasets::PortName::Fetcher,
                self.failing_phase.message(),
            ))
        } else {
            ctx.fetcher().fetch_bytes(&self.source, 1024)
        }
    }

    fn validate(
        &self,
        _ctx: &RecipeContext<'_>,
        fetched: Self::Fetched,
    ) -> Result<Self::Validated, RecipeError> {
        if matches!(self.failing_phase, FailingPhase::Validate) {
            Err(RecipeError::validate(self.failing_phase.message()))
        } else {
            Ok(fetched)
        }
    }

    fn prepare(
        &self,
        _ctx: &RecipeContext<'_>,
        validated: Self::Validated,
    ) -> Result<Self::Prepared, RecipeError> {
        if matches!(self.failing_phase, FailingPhase::Prepare) {
            Err(RecipeError::prepare(self.failing_phase.message()))
        } else {
            Ok(validated)
        }
    }

    fn publish(
        &self,
        _ctx: &RecipeContext<'_>,
        _prepared: Self::Prepared,
    ) -> Result<Self::Published, RecipeError> {
        if matches!(self.failing_phase, FailingPhase::Publish) {
            Err(RecipeError::publish(self.failing_phase.message()))
        } else {
            Ok(PublishedManifest::new(
                camino::Utf8PathBuf::from("manifests/failing.json"),
                ManifestDigest::zero_for_testing(),
            ))
        }
    }

    fn cleanup(&self, _ctx: &RecipeContext<'_>, partial: PartialState) -> Result<(), RecipeError> {
        *self.cleanup_state.lock().map_err(|_error| {
            RecipeError::cleanup(
                self.failing_phase.phase(),
                RecipeError::port(chutoro_bench_datasets::PortName::Storage, "lock poisoned"),
            )
        })? = Some(partial);
        Ok(())
    }
}
