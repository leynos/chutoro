//! Opaque key newtypes and recipe lifecycle contract types.
//!
//! Includes cache and object-store key wrappers, the manifest digest, and the
//! phase, port, and partial-state types used to describe recipe execution and
//! failure cleanup.
//!
//! Cache and object keys are opaque strings in the domain model. Any
//! interpretation of a key as a filesystem path, object-store prefix, or other
//! backend-specific address belongs to the [`Storage`](crate::Storage) and
//! [`Publisher`](crate::Publisher) adapters, so recipe logic never depends on
//! how a backend names its records.

use std::sync::Arc;

/// Cache object key used by mutable recipe storage.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CacheKey(Arc<str>);

/// Object-store key used by the publish sink.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ObjectKey(Arc<str>);

/// Implement the common opaque-key API for a recipe storage key type.
macro_rules! impl_opaque_key {
    ($name:ident, $create_doc:literal, $example_key:literal) => {
        impl $name {
            #[doc = $create_doc]
            ///
            /// The value is opaque to the domain. Interpreting it as a path or
            /// any other backend-specific address is the adapter's concern.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use chutoro_bench_datasets::", stringify!($name), ";")]
            ///
            #[doc = concat!("let key = ", stringify!($name), "::new(\"", $example_key, "\");")]
            #[doc = concat!("assert_eq!(key.as_ref(), \"", $example_key, "\");")]
            #[doc = concat!("assert_eq!(key.to_string(), \"", $example_key, "\");")]
            /// ```
            #[must_use]
            pub fn new(value: impl Into<Arc<str>>) -> Self {
                Self(value.into())
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.as_ref())
            }
        }
    };
}

impl_opaque_key!(
    CacheKey,
    "Create a cache key from an opaque identifier.",
    "mnist/raw.gz"
);
impl_opaque_key!(
    ObjectKey,
    "Create an object key from an opaque identifier.",
    "manifests/mnist.json"
);

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
///
/// The driver reports only the highest completed phase. It does not track
/// writes made through [`Storage`](crate::Storage) and supplies no cache keys,
/// so a recipe that must remove partial output records the [`CacheKey`] values
/// it wrote, or derives them deterministically from its inputs.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartialState {
    /// Highest phase completed before the failure.
    pub highest_completed_phase: Option<Phase>,
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
    /// ```
    #[must_use]
    pub const fn new(highest_completed_phase: Option<Phase>) -> Self {
        Self {
            highest_completed_phase,
        }
    }
}
