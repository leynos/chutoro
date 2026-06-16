//! Fetch port for source bytes.

use bytes::Bytes;

use crate::{RecipeError, SourceUrl};

/// Source-byte retrieval port.
pub trait Fetcher: Send + Sync {
    /// Fetch `url` and return the bytes read.
    ///
    /// Implementations must abort with [`RecipeError::FetchSizeExceeded`] when
    /// the response size exceeds `max_bytes`. The cap is mandatory; there is
    /// no path that returns more than `max_bytes` bytes successfully.
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError`] when the source cannot be read or exceeds the
    /// mandatory byte cap.
    fn fetch_bytes(&self, url: &SourceUrl, max_bytes: usize) -> Result<Bytes, RecipeError>;

    /// Fetch a sequence of URLs.
    ///
    /// The default implementation iterates serially in the declared order.
    fn fetch_many<'a>(
        &'a self,
        urls: &'a [(SourceUrl, usize)],
    ) -> Box<dyn Iterator<Item = (SourceUrl, Result<Bytes, RecipeError>)> + 'a> {
        Box::new(
            urls.iter()
                .map(|(url, max_bytes)| (url.clone(), self.fetch_bytes(url, *max_bytes))),
        )
    }
}
