//! Recipe execution context containing infrastructure ports.

use crate::{Fetcher, Publisher, Storage};

/// Borrowed port bundle supplied to recipe phases.
pub struct RecipeContext<'a> {
    fetcher: &'a dyn Fetcher,
    storage: &'a dyn Storage,
    publisher: &'a dyn Publisher,
}

impl<'a> RecipeContext<'a> {
    /// Create a recipe context from concrete port implementations.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use chutoro_bench_datasets::{
    ///     RecipeContext, RecipeError, SourceUrl,
    ///     testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage},
    /// };
    ///
    /// let source = SourceUrl::parse("https://example.test/data.bin")?;
    /// let fetcher = InMemoryFetcher::new([(source, Bytes::from_static(b"abc"))]);
    /// let storage = InMemoryStorage::default();
    /// let publisher = InMemoryPublisher::default();
    ///
    /// let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
    /// let _fetcher = ctx.fetcher();
    /// # Ok::<(), RecipeError>(())
    /// ```
    #[must_use]
    pub const fn new(
        fetcher: &'a dyn Fetcher,
        storage: &'a dyn Storage,
        publisher: &'a dyn Publisher,
    ) -> Self {
        Self {
            fetcher,
            storage,
            publisher,
        }
    }

    /// Borrow the fetcher port.
    #[must_use]
    pub const fn fetcher(&self) -> &'a dyn Fetcher {
        self.fetcher
    }

    /// Borrow the storage port.
    #[must_use]
    pub const fn storage(&self) -> &'a dyn Storage {
        self.storage
    }

    /// Borrow the publisher port.
    #[must_use]
    pub const fn publisher(&self) -> &'a dyn Publisher {
        self.publisher
    }
}
