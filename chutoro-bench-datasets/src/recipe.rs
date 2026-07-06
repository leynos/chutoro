//! Dataset recipe lifecycle contract.
//!
//! This module defines [`DatasetRecipe`], the trait for a typed dataset
//! lifecycle across four ordered phases: fetch, validate, prepare, and publish.
//! Phase outputs are represented by associated types and threaded into
//! subsequent phase method signatures, which prevents callers from skipping a
//! phase at compile time.

use crate::{
    DatasetInfo, PartialState, PublishedArtefact, RecipeContext, RecipeError, RecipeId,
    RecipeVersion, SourceSpec,
};

/// Fetch, validate, prepare, and publish a benchmark dataset.
///
/// The valid lifecycle passes each phase output to the next phase:
///
/// ```
/// # use chutoro_bench_datasets::{DatasetRecipe, RecipeContext, RecipeError};
/// fn run_lifecycle<R: DatasetRecipe>(
///     recipe: &R,
///     ctx: &RecipeContext<'_>,
/// ) -> Result<R::Published, RecipeError> {
///     let fetched = recipe.fetch(ctx)?;
///     let validated = recipe.validate(ctx, fetched)?;
///     let prepared = recipe.prepare(ctx, validated)?;
///     recipe.publish(ctx, prepared)
/// }
/// ```
pub trait DatasetRecipe: Send + Sync {
    /// Output of the fetch phase.
    type Fetched: Send + Sync;
    /// Output of the validate phase.
    type Validated: Send + Sync;
    /// Output of the prepare phase.
    type Prepared: Send + Sync;
    /// Output of the publish phase.
    type Published: PublishedArtefact;

    /// Return the stable recipe identifier.
    fn id(&self) -> RecipeId;

    /// Return the recipe version.
    fn version(&self) -> RecipeVersion;

    /// Return human-readable dataset metadata.
    fn info(&self) -> DatasetInfo;

    /// Return sources in the declared fetch order.
    fn sources(&self) -> &[SourceSpec];

    /// Fetch the recipe's source inputs.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when fetching through the context ports fails or
    /// the recipe rejects its source declaration.
    fn fetch(&self, ctx: &RecipeContext<'_>) -> Result<Self::Fetched, RecipeError>;

    /// Validate fetched inputs.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when the fetched inputs fail recipe-specific
    /// validation.
    fn validate(
        &self,
        ctx: &RecipeContext<'_>,
        fetched: Self::Fetched,
    ) -> Result<Self::Validated, RecipeError>;

    /// Prepare validated inputs for publication.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when preparation fails or an intermediate cache
    /// write cannot be completed.
    fn prepare(
        &self,
        ctx: &RecipeContext<'_>,
        validated: Self::Validated,
    ) -> Result<Self::Prepared, RecipeError>;

    /// Publish prepared artefacts.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when the publisher rejects or cannot persist
    /// the prepared artefacts.
    fn publish(
        &self,
        ctx: &RecipeContext<'_>,
        prepared: Self::Prepared,
    ) -> Result<Self::Published, RecipeError>;

    /// Roll back side effects of a partial run.
    ///
    /// Recipes whose prepare phase writes intermediates to [`crate::Storage`]
    /// should override this method to clean those entries up.
    /// [`crate::driver::run_recipe`] invokes cleanup only when a phase returns
    /// [`Err`]. It does not run cleanup when a phase panics, because phase
    /// execution currently attaches cleanup with `Result::map_err` and does
    /// not catch unwinds. Do not rely on this hook for panic recovery when
    /// rolling back partial storage writes.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when cleanup cannot remove recipe-owned partial
    /// state.
    fn cleanup(&self, _ctx: &RecipeContext<'_>, _partial: PartialState) -> Result<(), RecipeError> {
        Ok(())
    }
}
