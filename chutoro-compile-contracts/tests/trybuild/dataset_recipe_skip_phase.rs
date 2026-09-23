//! Compile-fail fixture proving `DatasetRecipe` phases cannot be skipped.

include!("ordered_recipe_support.rs");

fn main() -> Result<(), RecipeError> {
    let source = SourceUrl::parse("https://example.test/data.bin")?;
    let fetcher = InMemoryFetcher::new([(source.clone(), Bytes::from_static(b"abc"))]);
    let storage = InMemoryStorage::default();
    let publisher = InMemoryPublisher::default();
    let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
    let recipe = OrderedRecipe::new(source);

    let fetched = recipe.fetch(&ctx)?;
    let _prepared = recipe.prepare(&ctx, fetched)?;
    Ok(())
}
