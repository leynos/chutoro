//! Behavioural port-contract tests for dataset recipe fetchers.

use bytes::Bytes;
use camino::Utf8Path;
use cap_std::{ambient_authority, fs_utf8::Dir};
use chutoro_bench_datasets::{
    Fetcher, RecipeError, SourceUrl,
    testing::{FilesystemFetcher, InMemoryFetcher},
};
use rstest::fixture;
use rstest_bdd::StepResult;
use rstest_bdd_macros::{given, scenario, then, when};

#[derive(Default)]
struct PortWorld {
    fetcher: Option<Box<dyn Fetcher>>,
    tempdir: Option<tempfile::TempDir>,
    sources: Vec<SourceUrl>,
    fetched: Vec<Bytes>,
}

#[fixture]
fn world() -> PortWorld {
    PortWorld::default()
}

impl std::fmt::Debug for PortWorld {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PortWorld")
            .field("has_fetcher", &self.fetcher.is_some())
            .field("has_tempdir", &self.tempdir.is_some())
            .field("sources", &self.sources)
            .field("fetched", &self.fetched)
            .finish()
    }
}

#[given("the in-memory fetcher has two sources")]
fn in_memory_fetcher_has_two_sources(world: &mut PortWorld) -> StepResult<(), RecipeError> {
    let first = SourceUrl::parse("https://example.test/one.bin")?;
    let second = SourceUrl::parse("https://example.test/two.bin")?;
    world.fetcher = Some(Box::new(InMemoryFetcher::new([
        (first.clone(), Bytes::from_static(b"one")),
        (second.clone(), Bytes::from_static(b"two")),
    ])));
    world.sources = vec![first, second];
    Ok(())
}

#[given("the filesystem fetcher has two sources")]
fn filesystem_fetcher_has_two_sources(world: &mut PortWorld) -> StepResult<(), RecipeError> {
    let tempdir = tempfile::tempdir().map_err(|error| {
        RecipeError::port(chutoro_bench_datasets::PortName::Fetcher, error.to_string())
    })?;
    let root = Utf8Path::from_path(tempdir.path()).ok_or_else(|| {
        RecipeError::port(
            chutoro_bench_datasets::PortName::Fetcher,
            format!("non-UTF-8 path: {}", tempdir.path().display()),
        )
    })?;
    let fixture_dir = Dir::open_ambient_dir(root, ambient_authority()).map_err(|error| {
        RecipeError::port(chutoro_bench_datasets::PortName::Fetcher, error.to_string())
    })?;
    fixture_dir.write("one.bin", b"one").map_err(|error| {
        RecipeError::port(chutoro_bench_datasets::PortName::Fetcher, error.to_string())
    })?;
    fixture_dir.write("two.bin", b"two").map_err(|error| {
        RecipeError::port(chutoro_bench_datasets::PortName::Fetcher, error.to_string())
    })?;
    world.fetcher = Some(Box::new(FilesystemFetcher::new(root.to_path_buf())));
    world.tempdir = Some(tempdir);
    world.sources = vec![
        SourceUrl::parse("file://one.bin")?,
        SourceUrl::parse("file://two.bin")?,
    ];
    Ok(())
}

#[when("the recipe fetches the declared sources")]
fn recipe_fetches_declared_sources(world: &mut PortWorld) -> StepResult<(), RecipeError> {
    let fetcher = world.fetcher.as_deref().ok_or_else(|| {
        RecipeError::port(chutoro_bench_datasets::PortName::Fetcher, "missing fetcher")
    })?;
    world.fetched = world
        .sources
        .iter()
        .map(|source| fetcher.fetch_bytes(source, 3))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

#[then("the fetched bytes match the declared sources")]
fn fetched_bytes_match_declared_sources(world: &PortWorld) {
    assert_eq!(
        world.fetched,
        vec![Bytes::from_static(b"one"), Bytes::from_static(b"two")]
    );
}

#[scenario(
    path = "tests/features/recipe_ports.feature",
    name = "In-memory fetcher returns declared sources"
)]
fn in_memory_fetcher_returns_declared_sources(_world: PortWorld) {}

#[scenario(
    path = "tests/features/recipe_ports.feature",
    name = "Filesystem fetcher returns declared sources"
)]
fn filesystem_fetcher_returns_declared_sources(_world: PortWorld) {}
