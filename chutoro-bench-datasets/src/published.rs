//! Published artefact marker types.
//!
//! This module defines [`PublishedArtefact`], the sealed contract for values
//! returned by the recipe publish phase. The trait is sealed so this crate can
//! evolve the manifest contract when roadmap item `10.1.3` defines the
//! canonical schema, without downstream crates depending on an incomplete
//! shape. [`PublishedManifest`] is the minimal v1 handle: it carries only the
//! manifest URI and digest until richer metadata has a documented schema.

use camino::{Utf8Path, Utf8PathBuf};

use crate::ManifestDigest;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for values returned by the publish phase.
///
/// The trait is sealed so this crate can evolve the manifest contract when
/// roadmap item `10.1.3` defines the canonical schema.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "testing")]
/// # {
/// use camino::Utf8PathBuf;
/// use chutoro_bench_datasets::{
///     ManifestDigest, PublishedArtefact, PublishedManifest,
/// };
///
/// let manifest = PublishedManifest::new(
///     Utf8PathBuf::from("manifests/mnist.json"),
///     ManifestDigest::zero_for_testing(),
/// );
///
/// assert_eq!(manifest.manifest_uri().as_str(), "manifests/mnist.json");
/// assert_eq!(manifest.manifest_digest(), &ManifestDigest::zero_for_testing());
/// # }
/// ```
pub trait PublishedArtefact: sealed::Sealed + Send + Sync {
    /// Return the URI of the manifest describing the prepared dataset.
    fn manifest_uri(&self) -> &Utf8Path;

    /// Return the digest of the manifest bytes.
    fn manifest_digest(&self) -> &ManifestDigest;
}

/// Minimal published manifest handle used until the canonical schema lands.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishedManifest {
    manifest_uri: Utf8PathBuf,
    manifest_digest: ManifestDigest,
}

impl PublishedManifest {
    /// Create a published manifest handle.
    #[must_use]
    pub const fn new(manifest_uri: Utf8PathBuf, manifest_digest: ManifestDigest) -> Self {
        Self {
            manifest_uri,
            manifest_digest,
        }
    }
}

impl sealed::Sealed for PublishedManifest {}

impl PublishedArtefact for PublishedManifest {
    fn manifest_uri(&self) -> &Utf8Path {
        &self.manifest_uri
    }

    fn manifest_digest(&self) -> &ManifestDigest {
        &self.manifest_digest
    }
}
