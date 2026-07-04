//! Domain value types used by dataset recipes and ports.

use std::{fmt::Display, sync::Arc};

use camino::{Utf8Path, Utf8PathBuf};

use crate::RecipeError;

/// Stable identifier for a benchmark dataset recipe.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RecipeId(Arc<str>);

impl RecipeId {
    /// Create a recipe identifier from any string-like value.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeId;
    ///
    /// let id = RecipeId::new("mnist");
    /// assert_eq!(id.as_ref(), "mnist");
    /// ```
    #[must_use]
    pub fn new(value: impl Into<Arc<str>>) -> Self {
        Self(value.into())
    }
}

impl AsRef<str> for RecipeId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Display for RecipeId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_ref())
    }
}

/// SemVer-style recipe version.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecipeVersion {
    /// Major version component.
    pub major: u16,
    /// Minor version component.
    pub minor: u16,
    /// Patch version component.
    pub patch: u16,
}

impl RecipeVersion {
    /// Create a recipe version from explicit numeric components.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeVersion;
    ///
    /// assert_eq!(RecipeVersion::new(1, 2, 3).to_string(), "1.2.3");
    /// ```
    #[must_use]
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version in `<major>.<minor>.<patch>` form.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeVersion;
    ///
    /// let version = RecipeVersion::parse("1.2.3")?;
    /// assert_eq!(version.patch, 3);
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError::InvalidVersion`] when the value is not exactly
    /// three unsigned 16-bit integer components.
    #[must_use = "handle the parsed recipe version or its validation error"]
    pub fn parse(value: &str) -> Result<Self, RecipeError> {
        let mut parts = value.split('.');
        let major = parse_version_part(parts.next(), value)?;
        let minor = parse_version_part(parts.next(), value)?;
        let patch = parse_version_part(parts.next(), value)?;
        if parts.next().is_some() {
            return Err(RecipeError::invalid_version(value));
        }
        Ok(Self::new(major, minor, patch))
    }
}

impl Display for RecipeVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

fn parse_version_part(part: Option<&str>, value: &str) -> Result<u16, RecipeError> {
    part.ok_or_else(|| RecipeError::invalid_version(value))?
        .parse::<u16>()
        .map_err(|_error| RecipeError::invalid_version(value))
}

/// Supported source location for a recipe input.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SourceUrl(Arc<str>);

impl SourceUrl {
    /// Parse a source URL with a currently supported scheme.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::SourceUrl;
    ///
    /// let url = SourceUrl::parse("https://example.test/mnist.gz")?;
    /// assert_eq!(url.as_ref(), "https://example.test/mnist.gz");
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError::InvalidSource`] when the URL does not start with
    /// `https://`, `s3://`, or `file://`, or when the URL has no content after
    /// the scheme separator.
    #[must_use = "handle the parsed source URL or its validation error"]
    pub fn parse(value: &str) -> Result<Self, RecipeError> {
        if is_supported_source_scheme(value) && has_non_empty_source_remainder(value) {
            Ok(Self(Arc::from(value)))
        } else {
            Err(RecipeError::invalid_source(value))
        }
    }
}

impl AsRef<str> for SourceUrl {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Display for SourceUrl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_ref())
    }
}

fn is_supported_source_scheme(value: &str) -> bool {
    value.starts_with("https://") || value.starts_with("s3://") || value.starts_with("file://")
}

fn has_non_empty_source_remainder(value: &str) -> bool {
    value
        .split_once("://")
        .is_some_and(|(_scheme, remainder)| !remainder.is_empty())
}

/// Role played by a source within a dataset recipe.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SourceRole {
    /// Main source required for preparation.
    Primary,
    /// Backup or mirror source.
    Secondary,
    /// Additional source needed by the recipe.
    Auxiliary,
    /// Source containing nearest-neighbour or labelled ground truth.
    Groundtruth,
}

/// Declared source consumed by a dataset recipe.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SourceSpec {
    /// Location of the source.
    pub url: SourceUrl,
    /// Role of the source.
    pub role: SourceRole,
    /// Optional checksum placeholder, filled by roadmap item `10.1.2`.
    pub checksum: Option<Checksum>,
}

impl SourceSpec {
    /// Create a primary source specification.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{SourceSpec, SourceUrl};
    ///
    /// let source = SourceSpec::primary(SourceUrl::parse("file://mnist.bin")?);
    /// assert!(source.checksum.is_none());
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    #[must_use]
    pub const fn primary(url: SourceUrl) -> Self {
        Self {
            url,
            role: SourceRole::Primary,
            checksum: None,
        }
    }

    /// Create a source specification with an explicit role.
    #[must_use]
    pub const fn with_role(url: SourceUrl, role: SourceRole) -> Self {
        Self {
            url,
            role,
            checksum: None,
        }
    }
}

/// Cache object key used by mutable recipe storage.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CacheKey(Utf8PathBuf);

impl CacheKey {
    /// Create a cache key from a UTF-8 path.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::CacheKey;
    ///
    /// let key = CacheKey::new("mnist/raw.gz");
    /// assert_eq!(key.as_ref().as_str(), "mnist/raw.gz");
    /// ```
    #[must_use]
    pub fn new(value: impl Into<Utf8PathBuf>) -> Self {
        Self(value.into())
    }

    /// Return an owned copy of the wrapped path.
    #[must_use]
    pub fn to_path_buf(&self) -> Utf8PathBuf {
        self.0.clone()
    }

    /// Consume the key and return the wrapped path.
    #[must_use]
    pub fn into_inner(self) -> Utf8PathBuf {
        self.0
    }
}

impl AsRef<Utf8Path> for CacheKey {
    fn as_ref(&self) -> &Utf8Path {
        &self.0
    }
}

/// Object-store key used by the publish sink.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ObjectKey(Utf8PathBuf);

impl ObjectKey {
    /// Create an object key from a UTF-8 path.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::ObjectKey;
    ///
    /// let key = ObjectKey::new("manifests/mnist.json");
    /// assert_eq!(key.as_ref().as_str(), "manifests/mnist.json");
    /// ```
    #[must_use]
    pub fn new(value: impl Into<Utf8PathBuf>) -> Self {
        Self(value.into())
    }

    /// Return an owned copy of the wrapped path.
    #[must_use]
    pub fn to_path_buf(&self) -> Utf8PathBuf {
        self.0.clone()
    }

    /// Consume the key and return the wrapped path.
    #[must_use]
    pub fn into_inner(self) -> Utf8PathBuf {
        self.0
    }
}

impl AsRef<Utf8Path> for ObjectKey {
    fn as_ref(&self) -> &Utf8Path {
        &self.0
    }
}

/// Placeholder checksum type for future source validation.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Checksum {
    #[cfg(any())]
    /// Unreachable placeholder until roadmap item `10.1.2` selects hashing.
    Sha256([u8; 32]),
}

impl Checksum {
    /// Reject checksum parsing until roadmap item `10.1.2`.
    ///
    /// # Errors
    ///
    /// Always returns [`RecipeError::ChecksumUnsupported`] in this milestone.
    #[must_use = "handle the checksum parse result"]
    pub const fn parse(_value: &str) -> Result<Self, RecipeError> {
        Err(RecipeError::ChecksumUnsupported)
    }
}

/// Digest of the published manifest.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ManifestDigest([u8; 32]);

impl ManifestDigest {
    /// Return a zero digest for crate-provided testing artefacts.
    #[cfg(any(test, feature = "testing"))]
    #[must_use]
    pub const fn zero_for_testing() -> Self {
        Self([0; 32])
    }

    /// Return the digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Recipe lifecycle phase.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Phase {
    /// Fetch source inputs.
    Fetch,
    /// Validate fetched data.
    Validate,
    /// Prepare canonical benchmark artefacts.
    Prepare,
    /// Publish prepared artefacts.
    Publish,
}

/// I/O port involved in a recipe failure.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PortName {
    /// Source fetch port.
    Fetcher,
    /// Mutable cache port.
    Storage,
    /// Final artefact publish port.
    Publisher,
}

/// Partial recipe state passed to cleanup after a phase failure.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartialState {
    /// Highest phase completed before the failure.
    pub highest_completed_phase: Option<Phase>,
    /// Optional cache entry that may need removal by a recipe-specific cleanup.
    pub orphaned_cache_key: Option<CacheKey>,
}

impl PartialState {
    /// Create partial state with the highest completed phase.
    #[must_use]
    pub const fn new(highest_completed_phase: Option<Phase>) -> Self {
        Self {
            highest_completed_phase,
            orphaned_cache_key: None,
        }
    }
}
