//! Mutable cache port used by recipes during preparation.
//!
//! This module defines [`Storage`], the hexagonal port through which recipes
//! cache mutable intermediate artefacts during the prepare phase. Recipes access
//! the port through [`crate::RecipeContext`], so recipe code can describe when
//! bytes should be cached without depending on a concrete filesystem or object
//! store implementation. Storage is deliberately separate from
//! [`crate::Publisher`]: this port may overwrite intermediate values, while
//! publication models the final write-once dataset artefact boundary.

use bytes::Bytes;

use crate::{CacheKey, RecipeError};

/// Mutable cache for intermediate recipe artefacts.
///
/// Later writes overwrite earlier values. `Storage` has no version condition
/// and is intentionally distinct from [`crate::Publisher`], which models the
/// final write-once sink.
///
/// # Examples
///
/// ```
/// use bytes::Bytes;
/// use chutoro_bench_datasets::{
///     CacheKey, RecipeError, Storage,
///     testing::InMemoryStorage,
/// };
///
/// let storage = InMemoryStorage::default();
/// let key = CacheKey::new("cache/mnist/raw.gz");
///
/// assert_eq!(storage.get(&key)?, None);
/// storage.put(&key, b"first")?;
/// storage.put(&key, b"second")?;
/// assert_eq!(storage.get(&key)?, Some(Bytes::from_static(b"second")));
/// # Ok::<(), RecipeError>(())
/// ```
pub trait Storage: Send + Sync {
    /// Store bytes at `key`, overwriting any previous value.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when the cache cannot persist the bytes.
    fn put(&self, key: &CacheKey, bytes: &[u8]) -> Result<(), RecipeError>;

    /// Fetch bytes from `key`, returning `None` when the key is absent.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when the cache cannot be read.
    fn get(&self, key: &CacheKey) -> Result<Option<Bytes>, RecipeError>;
}
