//! Stub recipe used by crate consumers and integration tests.

use bytes::Bytes;
use camino::Utf8PathBuf;

use crate::{
    DatasetInfo, DatasetRecipe, ManifestDigest, ObjectKey, PublishedManifest, RecipeContext,
    RecipeError, RecipeId, RecipeVersion, SourceSpec, SourceUrl,
};

const DEFAULT_MAX_BYTES: usize = 1024 * 1024;

/// Simple recipe that fetches every source and publishes concatenated bytes.
#[derive(Clone, Debug)]
pub struct StubRecipe {
    id: RecipeId,
    version: RecipeVersion,
    sources: Vec<SourceSpec>,
    max_bytes: usize,
}

impl StubRecipe {
    /// Create a stub recipe with primary source specifications.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chutoro_bench_datasets::{SourceUrl, testing::StubRecipe};
    ///
    /// # fn main() -> Result<(), chutoro_bench_datasets::RecipeError> {
    /// let source = SourceUrl::parse("file://example-dataset.bin")?;
    /// let recipe = StubRecipe::new("example-dataset", vec![source]);
    ///
    /// assert_eq!(recipe.object_key().as_ref(), "prepared/example-dataset.bin");
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn new(id: impl Into<std::sync::Arc<str>>, sources: Vec<SourceUrl>) -> Self {
        Self {
            id: RecipeId::new(id),
            version: RecipeVersion::new(0, 1, 0),
            sources: sources.into_iter().map(SourceSpec::primary).collect(),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Set the per-source fetch byte cap.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{SourceUrl, testing::StubRecipe};
    ///
    /// let source = SourceUrl::parse("file://example-dataset.bin")?;
    /// let recipe = StubRecipe::new("example-dataset", vec![source]).with_max_bytes(128);
    ///
    /// assert_eq!(recipe.object_key().as_ref(), "prepared/example-dataset.bin");
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    #[must_use]
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Return the object key the stub recipe publishes to.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{SourceUrl, testing::StubRecipe};
    ///
    /// let source = SourceUrl::parse("file://example-dataset.bin")?;
    /// let recipe = StubRecipe::new("example-dataset", vec![source]);
    ///
    /// assert_eq!(recipe.object_key().as_ref(), "prepared/example-dataset.bin");
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    #[must_use]
    pub fn object_key(&self) -> ObjectKey {
        ObjectKey::new(format!("prepared/{}.bin", self.id.as_ref()))
    }
}

impl DatasetRecipe for StubRecipe {
    type Fetched = Vec<Bytes>;
    type Validated = Vec<Bytes>;
    type Prepared = Vec<Bytes>;
    type Published = PublishedManifest;

    fn id(&self) -> RecipeId {
        self.id.clone()
    }

    fn version(&self) -> RecipeVersion {
        self.version
    }

    fn info(&self) -> DatasetInfo {
        DatasetInfo::new(self.id(), self.version()).with_summary("stub benchmark dataset")
    }

    fn sources(&self) -> &[SourceSpec] {
        &self.sources
    }

    fn fetch(&self, ctx: &RecipeContext<'_>) -> Result<Self::Fetched, RecipeError> {
        self.sources
            .iter()
            .map(|source| ctx.fetcher().fetch_bytes(&source.url, self.max_bytes))
            .collect()
    }

    fn validate(
        &self,
        _ctx: &RecipeContext<'_>,
        fetched: Self::Fetched,
    ) -> Result<Self::Validated, RecipeError> {
        if fetched.is_empty() {
            Err(RecipeError::validate(
                "stub recipe requires at least one source",
            ))
        } else {
            Ok(fetched)
        }
    }

    fn prepare(
        &self,
        _ctx: &RecipeContext<'_>,
        validated: Self::Validated,
    ) -> Result<Self::Prepared, RecipeError> {
        Ok(validated)
    }

    fn publish(
        &self,
        ctx: &RecipeContext<'_>,
        prepared: Self::Prepared,
    ) -> Result<Self::Published, RecipeError> {
        let published_bytes = prepared.into_iter().flatten().collect::<Vec<_>>();
        let object_key = self.object_key();
        ctx.publisher().publish(&object_key, &published_bytes)?;
        Ok(PublishedManifest::new(
            Utf8PathBuf::from(format!("manifests/{}.json", self.id.as_ref())),
            ManifestDigest::zero_for_testing(),
        ))
    }
}
