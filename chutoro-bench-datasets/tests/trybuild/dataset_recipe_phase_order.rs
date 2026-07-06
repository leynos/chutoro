//! Compile-pass fixture for the ordered `DatasetRecipe` lifecycle.

include!("ordered_recipe_support.rs");

fn main() -> Result<(), RecipeError> {
    let source = SourceUrl::parse("https://example.test/data.bin")?;
    let fetcher = InMemoryFetcher::new([(source.clone(), Bytes::from_static(b"abc"))]);
    let storage = InMemoryStorage::default();
    let publisher = InMemoryPublisher::default();
    let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
    let recipe = OrderedRecipe::new(source);

    let fetched = recipe.fetch(&ctx)?;
    let validated = recipe.validate(&ctx, fetched)?;
    let prepared = recipe.prepare(&ctx, validated)?;
    let _published = recipe.publish(&ctx, prepared)?;
    Ok(())
}
