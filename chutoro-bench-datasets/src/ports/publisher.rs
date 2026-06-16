//! Final artefact publication port.
//!
//! This module defines [`Publisher`], the hexagonal port through which recipes
//! delegate final writes for prepared dataset artefacts. Recipes call
//! [`Publisher::publish`] through [`crate::RecipeContext`] during the publish
//! phase; the driver supplies concrete adapters such as object storage,
//! filesystem, or in-memory testing implementations. The separation keeps
//! recipe logic focused on dataset semantics while publication policy lives in
//! infrastructure.

use crate::{ObjectKey, RecipeError};

/// Write-once sink for prepared dataset artefacts.
///
/// Adapters added by roadmap item `10.1.4` may reject repeated writes with
/// optimistic concurrency checks. The in-memory testing adapter keeps the
/// latest value so tests can inspect what would have been published.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "testing")]
/// # {
/// use chutoro_bench_datasets::{
///     ObjectKey, Publisher, RecipeContext, RecipeError,
///     testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage},
/// };
///
/// # fn main() -> Result<(), RecipeError> {
/// let fetcher = InMemoryFetcher::default();
/// let storage = InMemoryStorage::default();
/// let publisher = InMemoryPublisher::default();
/// let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
/// let key = ObjectKey::new("prepared/mnist.bin");
///
/// ctx.publisher().publish(&key, b"prepared bytes")?;
/// assert_eq!(
///     publisher.records()?.get(&key),
///     Some(&bytes::Bytes::from_static(b"prepared bytes")),
/// );
/// # Ok(())
/// # }
/// # }
/// ```
pub trait Publisher: Send + Sync {
    /// Publish bytes at `key`.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when the sink rejects or cannot persist the
    /// published bytes.
    fn publish(&self, key: &ObjectKey, bytes: &[u8]) -> Result<(), RecipeError>;
}
