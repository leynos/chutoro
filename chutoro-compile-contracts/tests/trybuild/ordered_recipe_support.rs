// Shared ordered `DatasetRecipe` fixture used by the trybuild cases.

use bytes::Bytes;
use camino::Utf8PathBuf;
use chutoro_bench_datasets::{
    DatasetInfo, DatasetRecipe, ManifestDigest, PublishedManifest, RecipeContext, RecipeError,
    RecipeId, RecipeVersion, SourceSpec, SourceUrl,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage},
};

#[derive(Debug)]
struct Fetched(Bytes);

#[derive(Debug)]
struct Validated(Bytes);

#[derive(Debug)]
struct Prepared(Bytes);

#[derive(Debug)]
struct OrderedRecipe {
    source: SourceUrl,
    sources: Vec<SourceSpec>,
}

impl OrderedRecipe {
    fn new(source: SourceUrl) -> Self {
        Self {
            source: source.clone(),
            sources: vec![SourceSpec::primary(source)],
        }
    }
}

impl DatasetRecipe for OrderedRecipe {
    type Fetched = Fetched;
    type Validated = Validated;
    type Prepared = Prepared;
    type Published = PublishedManifest;

    fn id(&self) -> RecipeId {
        RecipeId::new("ordered")
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
        ctx.fetcher().fetch_bytes(&self.source, 1024).map(Fetched)
    }

    fn validate(
        &self,
        _ctx: &RecipeContext<'_>,
        fetched: Self::Fetched,
    ) -> Result<Self::Validated, RecipeError> {
        Ok(Validated(fetched.0))
    }

    fn prepare(
        &self,
        _ctx: &RecipeContext<'_>,
        validated: Self::Validated,
    ) -> Result<Self::Prepared, RecipeError> {
        Ok(Prepared(validated.0))
    }

    fn publish(
        &self,
        _ctx: &RecipeContext<'_>,
        prepared: Self::Prepared,
    ) -> Result<Self::Published, RecipeError> {
        let _bytes = prepared.0;
        Ok(PublishedManifest::new(
            Utf8PathBuf::from("manifests/ordered.json"),
            ManifestDigest::zero_for_testing(),
        ))
    }
}
