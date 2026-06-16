//! In-memory testing adapters for recipe ports.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bytes::Bytes;

use crate::{CacheKey, Fetcher, ObjectKey, PortName, Publisher, RecipeError, SourceUrl, Storage};

/// In-memory source fetcher.
#[derive(Clone, Debug, Default)]
pub struct InMemoryFetcher {
    sources: Arc<HashMap<SourceUrl, Bytes>>,
    requested: Arc<Mutex<Vec<SourceUrl>>>,
}

impl InMemoryFetcher {
    /// Create an in-memory fetcher from source-byte pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use chutoro_bench_datasets::{SourceUrl, testing::InMemoryFetcher};
    ///
    /// # fn main() -> Result<(), chutoro_bench_datasets::RecipeError> {
    /// let source = SourceUrl::parse("https://example.test/dataset.bin")?;
    /// let fetcher = InMemoryFetcher::new(vec![(source, Bytes::from_static(b"abc"))]);
    /// # let _ = fetcher;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn new(entries: impl IntoIterator<Item = (SourceUrl, Bytes)>) -> Self {
        Self {
            sources: Arc::new(entries.into_iter().collect()),
            requested: Arc::default(),
        }
    }

    /// Return URLs requested so far in observed order.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] if the request log lock is poisoned.
    pub fn requested_urls(&self) -> Result<Vec<SourceUrl>, RecipeError> {
        self.requested
            .lock()
            .map(|guard| guard.clone())
            .map_err(|_error| RecipeError::port(PortName::Fetcher, "request log lock poisoned"))
    }
}

impl Fetcher for InMemoryFetcher {
    fn fetch_bytes(&self, url: &SourceUrl, max_bytes: usize) -> Result<Bytes, RecipeError> {
        self.requested
            .lock()
            .map_err(|_error| RecipeError::port(PortName::Fetcher, "request log lock poisoned"))?
            .push(url.clone());
        let bytes = self
            .sources
            .get(url)
            .ok_or_else(|| RecipeError::port(PortName::Fetcher, format!("missing source {url}")))?;
        if bytes.len() > max_bytes {
            return Err(RecipeError::fetch_size_exceeded(url.clone(), max_bytes));
        }
        Ok(bytes.clone())
    }
}

/// In-memory mutable cache.
///
/// The adapter uses a [`Mutex`] to satisfy the public `Storage: Send + Sync`
/// contract. It is still intended only for deterministic in-process tests, not
/// cross-process cache coordination.
#[derive(Debug, Default)]
pub struct InMemoryStorage {
    records: Mutex<HashMap<CacheKey, Bytes>>,
}

impl InMemoryStorage {
    /// Consume the adapter and return stored records.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] if the storage lock is poisoned.
    pub fn into_records(self) -> Result<HashMap<CacheKey, Bytes>, RecipeError> {
        self.records
            .into_inner()
            .map_err(|_error| RecipeError::port(PortName::Storage, "storage lock poisoned"))
    }
}

impl Storage for InMemoryStorage {
    fn put(&self, key: &CacheKey, bytes: &[u8]) -> Result<(), RecipeError> {
        self.records
            .lock()
            .map_err(|_error| RecipeError::port(PortName::Storage, "storage lock poisoned"))?
            .insert(key.clone(), Bytes::copy_from_slice(bytes));
        Ok(())
    }

    fn get(&self, key: &CacheKey) -> Result<Option<Bytes>, RecipeError> {
        self.records
            .lock()
            .map(|records| records.get(key).cloned())
            .map_err(|_error| RecipeError::port(PortName::Storage, "storage lock poisoned"))
    }
}

/// In-memory publisher for final artefacts.
#[derive(Debug, Default)]
pub struct InMemoryPublisher {
    records: Mutex<HashMap<ObjectKey, Bytes>>,
}

impl InMemoryPublisher {
    /// Consume the adapter and return published records.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] if the publisher lock is poisoned.
    pub fn into_records(self) -> Result<HashMap<ObjectKey, Bytes>, RecipeError> {
        self.records
            .into_inner()
            .map_err(|_error| RecipeError::port(PortName::Publisher, "publisher lock poisoned"))
    }

    /// Return a cloned snapshot of the published records.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] if the publisher lock is poisoned.
    pub fn records(&self) -> Result<HashMap<ObjectKey, Bytes>, RecipeError> {
        self.records
            .lock()
            .map(|records| records.clone())
            .map_err(|_error| RecipeError::port(PortName::Publisher, "publisher lock poisoned"))
    }
}

impl Publisher for InMemoryPublisher {
    fn publish(&self, key: &ObjectKey, bytes: &[u8]) -> Result<(), RecipeError> {
        self.records
            .lock()
            .map_err(|_error| RecipeError::port(PortName::Publisher, "publisher lock poisoned"))?
            .insert(key.clone(), Bytes::copy_from_slice(bytes));
        Ok(())
    }
}
