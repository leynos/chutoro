//! Dataset recipe contracts for benchmark dataset preparation.
//!
//! `chutoro-bench-datasets` defines the trait surface used by future
//! benchmark dataset recipes. A recipe moves through four ordered phases:
//! fetch obtains source bytes through a [`Fetcher`], validate proves those
//! bytes are suitable for preparation, prepare converts validated input into a
//! benchmark-ready intermediate, and publish writes the final artefact through
//! a [`Publisher`].
//!
//! The type returned by each phase is consumed by the next phase. This
//! typestate-style handoff means callers cannot accidentally publish fetched
//! but unvalidated bytes.
//!
//! Deferred roadmap items intentionally remain out of this crate. Network
//! download primitives, extractor ports, canonical manifest schemas, object
//! store adapters, lockfiles, provenance gates, licence gates, and
//! property-based metadata verification are added by later milestones.
//!
//! # Example
//!
//! The `testing` feature exposes a complete in-memory recipe and adapters:
//!
//! ```rust
//! # #[cfg(feature = "testing")]
//! # {
//! use bytes::Bytes;
//! use chutoro_bench_datasets::testing::{
//!     InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe,
//! };
//! use chutoro_bench_datasets::{PublishedArtefact, RecipeContext, SourceUrl, run_recipe};
//!
//! let source = SourceUrl::parse("https://example.test/dataset.bin")?;
//! let fetcher = InMemoryFetcher::new([(source.clone(), Bytes::from_static(b"abc"))]);
//! let storage = InMemoryStorage::default();
//! let publisher = InMemoryPublisher::default();
//! let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
//! let recipe = StubRecipe::new("example", vec![source]);
//!
//! let published = run_recipe(&recipe, &ctx)?;
//! assert_eq!(published.manifest_uri().as_str(), "manifests/example.json");
//! # }
//! # Ok::<(), chutoro_bench_datasets::RecipeError>(())
//! ```
//!
//! # Concurrency
//!
//! Concurrent invocations for the same [`RecipeId`] may produce
//! non-deterministic cache results until roadmap item `10.1.5` introduces
//! lockfile semantics. Recipes and ports are `Send + Sync` so a future driver
//! can coordinate work explicitly, but this milestone provides no cache-level
//! mutual exclusion.

mod context;
mod driver;
mod error;
mod info;
mod newtypes;
mod ports;
mod published;
mod recipe;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use context::RecipeContext;
pub use driver::run_recipe;
pub use error::{FetchSizeExceeded, PortFailure, RecipeError};
pub use info::DatasetInfo;
pub use newtypes::{
    CacheKey, Checksum, ManifestDigest, ObjectKey, PartialState, Phase, PortName, RecipeId,
    RecipeVersion, SourceRole, SourceSpec, SourceUrl,
};
pub use ports::{Fetcher, Publisher, Storage};
pub use published::{PublishedArtefact, PublishedManifest};
pub use recipe::DatasetRecipe;
