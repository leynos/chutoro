//! Unit tests for the dataset recipe lifecycle driver.

use bytes::Bytes;
use chutoro_bench_datasets::{
    DatasetInfo, DatasetRecipe, FetchSizeExceeded, ManifestDigest, ObjectKey, PartialState,
    PublishedArtefact, PublishedManifest, RecipeContext, RecipeError, RecipeId, RecipeVersion,
    SourceSpec, SourceUrl, run_recipe,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe},
};
use rstest::{fixture, rstest};
use tracing_test::traced_test;

#[fixture]
fn source() -> SourceUrl {
    match SourceUrl::parse("https://example.test/data.bin") {
        Ok(url) => url,
        Err(error) => panic!("test source URL should parse: {error}"),
    }
}

#[fixture]
fn fetcher(source: SourceUrl) -> InMemoryFetcher {
    InMemoryFetcher::new([(source, Bytes::from_static(b"abc"))])
}

#[fixture]
fn storage() -> InMemoryStorage {
    InMemoryStorage::default()
}

#[fixture]
fn publisher() -> InMemoryPublisher {
    InMemoryPublisher::default()
}

struct RecipeSetup {
    fetcher: InMemoryFetcher,
    storage: InMemoryStorage,
    publisher: InMemoryPublisher,
}

impl RecipeSetup {
    fn context(&self) -> RecipeContext<'_> {
        RecipeContext::new(&self.fetcher, &self.storage, &self.publisher)
    }
}

#[fixture]
fn ctx(
    fetcher: InMemoryFetcher,
    storage: InMemoryStorage,
    publisher: InMemoryPublisher,
) -> RecipeSetup {
    RecipeSetup {
        fetcher,
        storage,
        publisher,
    }
}

#[traced_test]
#[rstest]
fn run_recipe_publishes_prepared_bytes(source: SourceUrl, ctx: RecipeSetup) {
    let recipe = StubRecipe::new("stub", vec![source]);
    let context = ctx.context();

    let artefact = run_recipe(&recipe, &context).expect("stub recipe should run");

    assert_eq!(artefact.manifest_uri().as_str(), "manifests/stub.json");
    assert!(logs_contain("executing dataset recipe phase"));
    let records = ctx
        .publisher
        .records()
        .expect("published records should be readable");
    assert_eq!(
        records.get(&ObjectKey::new("prepared/stub.bin")),
        Some(&Bytes::from_static(b"abc")),
    );
}

#[rstest]
fn fetch_size_limit_is_propagated(source: SourceUrl, ctx: RecipeSetup) {
    let recipe = StubRecipe::new("stub", vec![source]).with_max_bytes(2);
    let context = ctx.context();

    let Err(error) = run_recipe(&recipe, &context) else {
        panic!("oversized source should fail");
    };

    assert!(matches!(
        error,
        RecipeError::FetchSizeExceeded(FetchSizeExceeded { limit_bytes: 2, .. })
    ));
}

#[rstest]
#[case::fetch(FailingPhase::Fetch, None)]
#[case::validate(FailingPhase::Validate, Some(chutoro_bench_datasets::Phase::Fetch))]
#[case::prepare(FailingPhase::Prepare, Some(chutoro_bench_datasets::Phase::Validate))]
#[case::publish(FailingPhase::Publish, Some(chutoro_bench_datasets::Phase::Prepare))]
fn failure_invokes_cleanup_with_partial_state(
    source: SourceUrl,
    ctx: RecipeSetup,
    #[case] failing_phase: FailingPhase,
    #[case] expected_completed_phase: Option<chutoro_bench_datasets::Phase>,
) {
    let recipe = FailingRecipe::new(source, failing_phase);
    let context = ctx.context();

    let Err(error) = run_recipe(&recipe, &context) else {
        panic!("configured phase should fail");
    };

    assert!(error.to_string().contains(failing_phase.message()));
    assert_eq!(
        recipe.cleanup_state(),
        Some(PartialState::new(expected_completed_phase)),
    );
}

#[rstest]
fn cleanup_failure_reports_failed_phase_and_cleanup_source(source: SourceUrl, ctx: RecipeSetup) {
    let recipe =
        FailingRecipe::new(source, FailingPhase::Validate).with_cleanup_error("cleanup failed");
    let context = ctx.context();

    let Err(error) = run_recipe(&recipe, &context) else {
        panic!("cleanup failure should replace the original phase error");
    };

    let RecipeError::Cleanup {
        phase,
        source: cleanup_source,
    } = error
    else {
        panic!("cleanup failure should return RecipeError::Cleanup");
    };
    assert_eq!(phase, chutoro_bench_datasets::Phase::Validate);
    assert!(cleanup_source.to_string().contains("cleanup failed"));
    assert_eq!(
        recipe.cleanup_state(),
        Some(PartialState::new(Some(
            chutoro_bench_datasets::Phase::Fetch
        ))),
    );
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
    cleanup_error: Option<&'static str>,
    cleanup_state: std::sync::Mutex<Option<PartialState>>,
}

impl FailingRecipe {
    fn new(source: SourceUrl, failing_phase: FailingPhase) -> Self {
        Self {
            sources: vec![SourceSpec::primary(source.clone())],
            source,
            failing_phase,
            cleanup_error: None,
            cleanup_state: std::sync::Mutex::new(None),
        }
    }

    const fn with_cleanup_error(mut self, cleanup_error: &'static str) -> Self {
        self.cleanup_error = Some(cleanup_error);
        self
    }

    fn cleanup_state(&self) -> Option<PartialState> {
        match self.cleanup_state.lock() {
            Ok(state) => state.clone(),
            Err(error) => panic!("cleanup state lock should not be poisoned: {error}"),
        }
    }

    fn fail_or_pass<T>(
        &self,
        target: FailingPhase,
        make_error: impl FnOnce(&'static str) -> RecipeError,
        value: T,
    ) -> Result<T, RecipeError> {
        if self.failing_phase == target {
            Err(make_error(self.failing_phase.message()))
        } else {
            Ok(value)
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
        self.fail_or_pass(FailingPhase::Validate, RecipeError::validate, fetched)
    }

    fn prepare(
        &self,
        _ctx: &RecipeContext<'_>,
        validated: Self::Validated,
    ) -> Result<Self::Prepared, RecipeError> {
        self.fail_or_pass(FailingPhase::Prepare, RecipeError::prepare, validated)
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
        self.cleanup_error.map_or_else(
            || Ok(()),
            |cleanup_error| {
                Err(RecipeError::port(
                    chutoro_bench_datasets::PortName::Storage,
                    cleanup_error,
                ))
            },
        )
    }
}
