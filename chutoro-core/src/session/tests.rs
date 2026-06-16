//! Unit tests for [`ClusteringSession`] and its supporting session scaffolding,
//! defined in `chutoro-core/src/session/mod.rs`.
//!
//! These are white-box unit tests that live inside the session module and use
//! [`rstest`] fixtures (`session_builder`) to reduce per-test boilerplate.
//! They sit at the base of the testing hierarchy and cover the `append` API
//! at the method level — happy paths, validation failures, partial-progress
//! semantics, and property-based invariants via [`proptest`].
//!
//! [`ClusteringSession`]: super::ClusteringSession

mod append;
mod builder;
mod common;
mod concurrency;
mod core_distance;
#[cfg(feature = "metrics")]
mod metrics;
mod properties;
