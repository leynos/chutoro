//! Driver for executing dataset recipes.

use tracing::{info, info_span, warn};

use crate::{DatasetRecipe, PartialState, Phase, RecipeContext, RecipeError};

/// Run all recipe phases in order.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "testing")]
/// # {
/// use bytes::Bytes;
/// use chutoro_bench_datasets::testing::{
///     InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe,
/// };
/// use chutoro_bench_datasets::{PublishedArtefact, RecipeContext, SourceUrl, run_recipe};
///
/// # fn main() -> Result<(), chutoro_bench_datasets::RecipeError> {
/// let source = SourceUrl::parse("https://example.test/dataset.bin")?;
/// let fetcher = InMemoryFetcher::new([(source.clone(), Bytes::from_static(b"abc"))]);
/// let storage = InMemoryStorage::default();
/// let publisher = InMemoryPublisher::default();
/// let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
/// let recipe = StubRecipe::new("example", vec![source]);
///
/// let published = run_recipe(&recipe, &ctx)?;
/// assert_eq!(published.manifest_uri().as_str(), "manifests/example.json");
/// # Ok(())
/// # }
/// # }
/// ```
///
/// # Errors
///
/// Returns the first phase error. If cleanup also fails, returns a cleanup
/// error describing the failed phase.
pub fn run_recipe<R: DatasetRecipe>(
    recipe: &R,
    ctx: &RecipeContext<'_>,
) -> Result<R::Published, RecipeError> {
    let fetched = execute_phase(
        recipe,
        ctx,
        PhaseExecution::new(None, Phase::Fetch),
        DatasetRecipe::fetch,
    )?;
    let validated = execute_phase(
        recipe,
        ctx,
        PhaseExecution::new(Some(Phase::Fetch), Phase::Validate),
        |active_recipe, recipe_ctx| active_recipe.validate(recipe_ctx, fetched),
    )?;
    let prepared = execute_phase(
        recipe,
        ctx,
        PhaseExecution::new(Some(Phase::Validate), Phase::Prepare),
        |active_recipe, recipe_ctx| active_recipe.prepare(recipe_ctx, validated),
    )?;
    execute_phase(
        recipe,
        ctx,
        PhaseExecution::new(Some(Phase::Prepare), Phase::Publish),
        |active_recipe, recipe_ctx| active_recipe.publish(recipe_ctx, prepared),
    )
}

fn execute_phase<R, T>(
    recipe: &R,
    ctx: &RecipeContext<'_>,
    execution: PhaseExecution,
    run: impl FnOnce(&R, &RecipeContext<'_>) -> Result<T, RecipeError>,
) -> Result<T, RecipeError>
where
    R: DatasetRecipe,
{
    info_span!("dataset_recipe_phase", phase = ?execution.phase)
        .in_scope(|| {
            let recipe_id = recipe.id();
            info!(recipe_id = %recipe_id.as_ref(), "executing dataset recipe phase");
            run(recipe, ctx)
        })
        .map_err(|error| {
            cleanup_after_error(
                recipe,
                ctx,
                PhaseFailure::new(execution.highest_completed_phase, execution.phase, error),
            )
        })
}

#[derive(Clone, Copy, Debug)]
struct PhaseExecution {
    highest_completed_phase: Option<Phase>,
    phase: Phase,
}

impl PhaseExecution {
    const fn new(highest_completed_phase: Option<Phase>, phase: Phase) -> Self {
        Self {
            highest_completed_phase,
            phase,
        }
    }
}

#[derive(Debug)]
struct PhaseFailure {
    highest_completed_phase: Option<Phase>,
    failed_phase: Phase,
    original: RecipeError,
}

impl PhaseFailure {
    const fn new(
        highest_completed_phase: Option<Phase>,
        failed_phase: Phase,
        original: RecipeError,
    ) -> Self {
        Self {
            highest_completed_phase,
            failed_phase,
            original,
        }
    }
}

fn cleanup_after_error<R: DatasetRecipe>(
    recipe: &R,
    ctx: &RecipeContext<'_>,
    failure: PhaseFailure,
) -> RecipeError {
    let partial = PartialState::new(failure.highest_completed_phase);
    match recipe.cleanup(ctx, partial) {
        Ok(()) => failure.original,
        Err(cleanup_error) => {
            warn!(
                phase = ?failure.failed_phase,
                original_error = %failure.original,
                cleanup_error = %cleanup_error,
                "dataset recipe phase failed and cleanup also failed",
            );
            RecipeError::cleanup(failure.failed_phase, cleanup_error)
        }
    }
}
