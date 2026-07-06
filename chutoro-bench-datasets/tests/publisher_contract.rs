//! Integration tests for the publisher port contract.

use bytes::Bytes;
use chutoro_bench_datasets::{
    ObjectKey, RecipeContext,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage},
};

#[test]
fn recipe_context_publisher_publishes_prepared_bytes() {
    let fetcher = InMemoryFetcher::default();
    let storage = InMemoryStorage::default();
    let publisher = InMemoryPublisher::default();
    let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
    let key = ObjectKey::new("prepared/example.bin");

    ctx.publisher()
        .publish(&key, b"prepared bytes")
        .expect("publisher should persist prepared bytes");

    assert_eq!(
        publisher
            .records()
            .expect("publisher records should be readable")
            .get(&key),
        Some(&Bytes::from_static(b"prepared bytes")),
    );
}
