//! Filesystem-backed fetcher used by port contract tests.

use std::io::Read;

use bytes::Bytes;
use camino::{Utf8Path, Utf8PathBuf};
use cap_std::{ambient_authority, fs_utf8::Dir};

use crate::{Fetcher, PortName, RecipeError, SourceUrl};

/// Fetcher that reads `file://` URLs relative to a fixture root.
#[derive(Clone, Debug)]
pub struct FilesystemFetcher {
    root: Utf8PathBuf,
}

impl FilesystemFetcher {
    /// Create a filesystem fetcher rooted at `root`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use camino::Utf8PathBuf;
    /// use chutoro_bench_datasets::testing::FilesystemFetcher;
    ///
    /// let fixture_root = Utf8PathBuf::from("fixtures/datasets");
    /// let fetcher = FilesystemFetcher::new(fixture_root);
    /// # let _ = fetcher;
    /// ```
    #[must_use]
    pub fn new(root: impl Into<Utf8PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Fetcher for FilesystemFetcher {
    fn fetch_bytes(&self, url: &SourceUrl, max_bytes: usize) -> Result<Bytes, RecipeError> {
        let relative = relative_file_path(url)?;
        let root = Dir::open_ambient_dir(&self.root, ambient_authority())
            .map_err(|error| RecipeError::port(PortName::Fetcher, error.to_string()))?;
        let mut file = root
            .open(relative)
            .map_err(|error| RecipeError::port(PortName::Fetcher, error.to_string()))?;
        let limit = max_bytes as u64;
        let file_len = file
            .metadata()
            .map_err(|error| RecipeError::port(PortName::Fetcher, error.to_string()))?
            .len();
        if file_len > limit {
            return Err(RecipeError::fetch_size_exceeded(url.clone(), max_bytes));
        }
        let capacity = usize::try_from(file_len)
            .map_err(|error| RecipeError::port(PortName::Fetcher, error.to_string()))?;
        let mut buffer = Vec::with_capacity(capacity);
        file.by_ref()
            .take(limit.saturating_add(1))
            .read_to_end(&mut buffer)
            .map_err(|error| RecipeError::port(PortName::Fetcher, error.to_string()))?;
        if buffer.len() > max_bytes {
            return Err(RecipeError::fetch_size_exceeded(url.clone(), max_bytes));
        }
        Ok(Bytes::from(buffer))
    }
}

fn relative_file_path(url: &SourceUrl) -> Result<&Utf8Path, RecipeError> {
    let value = url.as_ref();
    let relative = value
        .strip_prefix("file://")
        .ok_or_else(|| RecipeError::invalid_source(value))?;
    let path = Utf8Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| component.as_str() == "..")
    {
        return Err(RecipeError::invalid_source(value));
    }
    Ok(path)
}
