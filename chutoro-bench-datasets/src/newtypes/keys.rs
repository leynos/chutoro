//! Path-backed key newtypes and recipe lifecycle contract types.
//!
//! Includes cache and object-store key wrappers, checksum and manifest
//! digest placeholders, and the phase, port, and partial-state types used
//! to describe recipe execution and failure cleanup.

use camino::{Utf8Path, Utf8PathBuf};

use crate::RecipeError;

/// Cache object key used by mutable recipe storage.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CacheKey(Utf8PathBuf);

/// Object-store key used by the publish sink.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ObjectKey(Utf8PathBuf);

macro_rules! impl_path_key {
    ($name:ident, $create_doc:literal, $example_path:literal) => {
        impl $name {
            #[doc = $create_doc]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use chutoro_bench_datasets::", stringify!($name), ";")]
            ///
            #[doc = concat!("let key = ", stringify!($name), "::new(\"", $example_path, "\");")]
            #[doc = concat!("assert_eq!(key.as_ref().as_str(), \"", $example_path, "\");")]
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

        impl AsRef<Utf8Path> for $name {
            fn as_ref(&self) -> &Utf8Path {
                &self.0
            }
        }
    };
}

impl_path_key!(
    CacheKey,
    "Create a cache key from a UTF-8 path.",
    "mnist/raw.gz"
);
impl_path_key!(
    ObjectKey,
    "Create an object key from a UTF-8 path.",
    "manifests/mnist.json"
);

/// Placeholder checksum type for future source validation.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Checksum {
    /// Unreachable placeholder until roadmap item `10.1.2` selects hashing.
    #[cfg(any())]
    Sha256([u8; 32]),
}

impl Checksum {
    /// Reject checksum parsing until roadmap item `10.1.2`.
    ///
    /// # Errors
    ///
    /// Always returns [`RecipeError::ChecksumUnsupported`] in this milestone.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{Checksum, RecipeError};
    ///
    /// let error = Checksum::parse("sha256:abc123")
    ///     .expect_err("checksums are deferred to roadmap item 10.1.2");
    /// assert!(matches!(error, RecipeError::ChecksumUnsupported));
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{PartialState, Phase};
    ///
    /// let state = PartialState::new(Some(Phase::Validate));
    /// assert_eq!(state.highest_completed_phase, Some(Phase::Validate));
    /// assert!(state.orphaned_cache_key.is_none());
    /// ```
    #[must_use]
    pub const fn new(highest_completed_phase: Option<Phase>) -> Self {
        Self {
            highest_completed_phase,
            orphaned_cache_key: None,
        }
    }
}
