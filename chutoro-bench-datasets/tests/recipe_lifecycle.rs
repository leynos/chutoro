//! Unit tests for the dataset recipe lifecycle driver.

use std::error::Error;

use bytes::Bytes;
use chutoro_bench_datasets::{
    CleanupFailure, DatasetInfo, DatasetRecipe, FetchSizeExceeded, ManifestDigest, ObjectKey,
    PartialState, Phase, PortName, PublishedArtefact, PublishedManifest, RecipeContext,
    RecipeError, RecipeId, RecipeVersion, SourceSpec, SourceUrl, run_recipe,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe},
};
use rstest::{fixture, rstest};
use tracing_test::traced_test;

mod common;
use common::expect_err;

#[fixture]
fn source() -> Result<SourceUrl, RecipeError> {
    SourceUrl::parse("https://example.test/data.bin")
}

#[fixture]
fn fetcher(source: Result<SourceUrl, RecipeError>) -> Result<InMemoryFetcher, RecipeError> {
    Ok(InMemoryFetcher::new([(
        source?,
        Bytes::from_static(b"abc"),
    )]))
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
    fetcher: Result<InMemoryFetcher, RecipeError>,
    storage: InMemoryStorage,
    publisher: InMemoryPublisher,
) -> Result<RecipeSetup, RecipeError> {
    Ok(RecipeSetup {
        fetcher: fetcher?,
        storage,
        publisher,
    })
}

#[traced_test]
#[rstest]
fn run_recipe_publishes_prepared_bytes(
    #[from(source)] source_res: Result<SourceUrl, RecipeError>,
    #[from(ctx)] ctx_res: Result<RecipeSetup, RecipeError>,
) {
    let source = source_res.expect("test source URL should parse");
    let ctx = ctx_res.expect("recipe setup should build");
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
fn fetch_size_limit_is_propagated(
    #[from(source)] source_res: Result<SourceUrl, RecipeError>,
    #[from(ctx)] ctx_res: Result<RecipeSetup, RecipeError>,
) {
    let source = source_res.expect("test source URL should parse");
    let ctx = ctx_res.expect("recipe setup should build");
    let recipe = StubRecipe::new("stub", vec![source]).with_max_bytes(2);
    let context = ctx.context();

    let error = expect_err!(
        run_recipe(&recipe, &context),
        "oversized source should fail"
    );

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
    #[from(source)] source_res: Result<SourceUrl, RecipeError>,
    #[from(ctx)] ctx_res: Result<RecipeSetup, RecipeError>,
    #[case] failing_phase: FailingPhase,
    #[case] expected_completed_phase: Option<chutoro_bench_datasets::Phase>,
) {
    let source = source_res.expect("test source URL should parse");
    let ctx = ctx_res.expect("recipe setup should build");
    let recipe = FailingRecipe::new(source, failing_phase);
    let context = ctx.context();

    let error = expect_err!(
        run_recipe(&recipe, &context),
        "configured phase should fail"
    );

    assert!(
        !matches!(&error, RecipeError::Cleanup { .. }),
        "successful cleanup should preserve the original phase error",
    );
    assert!(error.to_string().contains(failing_phase.message()));
    let recorded = recipe
        .cleanup_state()
        .expect("cleanup state lock should not be poisoned")
        .expect("cleanup should receive partial state");
    assert_eq!(recorded.highest_completed_phase, expected_completed_phase);
}

/// Reason recorded by [`FailingRecipe`] when its cleanup hook fails.
const CLEANUP_REASON: &str = "cleanup failed";

#[rstest]
#[case::fetch(FailingPhase::Fetch, None)]
#[case::validate(FailingPhase::Validate, Some(Phase::Fetch))]
#[case::prepare(FailingPhase::Prepare, Some(Phase::Validate))]
#[case::publish(FailingPhase::Publish, Some(Phase::Prepare))]
fn cleanup_failure_reports_failed_phase_and_cleanup_source(
    #[from(source)] source_res: Result<SourceUrl, RecipeError>,
    #[from(ctx)] ctx_res: Result<RecipeSetup, RecipeError>,
    #[case] failing_phase: FailingPhase,
    #[case] expected_completed_phase: Option<Phase>,
) {
    let source = source_res.expect("test source URL should parse");
    let ctx = ctx_res.expect("recipe setup should build");
    let recipe = FailingRecipe::new(source, failing_phase).with_cleanup_error(CLEANUP_REASON);
    let context = ctx.context();

    let error = expect_err!(
        run_recipe(&recipe, &context),
        "a failing phase and failing cleanup should both be reported"
    );

    let RecipeError::Cleanup { phase, .. } = &error else {
        panic!("cleanup failure should return RecipeError::Cleanup");
    };
    assert_eq!(*phase, failing_phase.phase());

    let cleanup_failure = error
        .cleanup_failure()
        .expect("cleanup should retain both failures");
    assert!(
        failing_phase.matches_phase_error(cleanup_failure.phase_error()),
        "phase error should be {}, got {}",
        failing_phase.message(),
        cleanup_failure.phase_error(),
    );
    let RecipeError::Port(cleanup_error) = cleanup_failure.cleanup_error() else {
        panic!("cleanup hook should report a storage port failure");
    };
    assert_eq!(cleanup_error.port, PortName::Storage);
    assert_eq!(cleanup_error.reason.as_ref(), CLEANUP_REASON);

    let chained_failure = error
        .source()
        .expect("cleanup error should expose its source")
        .downcast_ref::<CleanupFailure>()
        .expect("cleanup source should be the driver payload");
    let chained_phase_error = chained_failure
        .source()
        .expect("cleanup payload should expose the phase error")
        .downcast_ref::<RecipeError>()
        .expect("cleanup payload source should be the phase recipe error");
    assert!(
        failing_phase.matches_phase_error(chained_phase_error),
        "chained phase error should be {}, got {chained_phase_error}",
        failing_phase.message(),
    );

    let message = error.to_string();
    assert!(
        message.contains(failing_phase.message()),
        "outer error should retain the phase reason: {message}",
    );
    assert!(
        message.contains(CLEANUP_REASON),
        "outer error should retain the cleanup reason: {message}",
    );

    let recorded = recipe
        .cleanup_state()
        .expect("cleanup state lock should not be poisoned")
        .expect("cleanup should receive partial state");
    assert_eq!(recorded.highest_completed_phase, expected_completed_phase);
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

    const fn phase(self) -> Phase {
        match self {
            Self::Fetch => Phase::Fetch,
            Self::Validate => Phase::Validate,
            Self::Prepare => Phase::Prepare,
            Self::Publish => Phase::Publish,
        }
    }

    /// Returns whether `error` is the exact [`RecipeError`] this phase emits.
    fn matches_phase_error(self, error: &RecipeError) -> bool {
        let message = self.message();
        match (self, error) {
            (Self::Fetch, RecipeError::Port(failure)) => {
                failure.port == PortName::Fetcher && failure.reason.as_ref() == message
            }
            (Self::Validate, RecipeError::Validate(reason))
            | (Self::Prepare, RecipeError::Prepare(reason))
            | (Self::Publish, RecipeError::Publish(reason)) => reason.as_ref() == message,
            _ => false,
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

    /// Reads the recorded cleanup state, propagating a mutex poison error
    /// rather than panicking so callers decide how to report the failure.
    fn cleanup_state(&self) -> Result<Option<PartialState>, String> {
        self.cleanup_state
            .lock()
            .map(|state| state.clone())
            .map_err(|error| format!("cleanup state lock should not be poisoned: {error}"))
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
// weave: run 'weave explain chutoro-bench-datasets/tests/recipe_lifecycle.rs' for per-hunk detail, 'weave check' to verify your resolution
