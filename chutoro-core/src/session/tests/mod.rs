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

use crate::{DataSource, DataSourceError, MetricDescriptor};

mod append;
mod builder;
mod common;
mod concurrency;
mod core_distance;
mod core_distance_errors;
mod core_distance_support;
// The failure-injecting source is shared by the core-distance and refresh
// error suites. It is declared once here; per-suite declarations would load
// the same file as several modules, which Clippy rejects.
#[path = "../../../tests/support/failable_source.rs"]
mod failable_source;
#[cfg(feature = "metrics")]
mod metrics;
mod properties;
mod refresh;
mod refresh_errors;
#[cfg(feature = "metrics")]
mod refresh_metrics;
mod refresh_properties;
