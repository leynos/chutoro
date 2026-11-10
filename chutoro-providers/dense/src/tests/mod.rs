//! Dense provider test suite covering errors, ingestion, providers, sources, and shared fixtures.
//!
//! Reusable Arrow and Parquet builders live in the crate's feature-gated
//! `testing` module; `support` retains only provider-specific test helpers.
pub(crate) use super::{DenseMatrixProvider, DenseMatrixProviderError, DenseSource};

mod errors;
mod ingest;
mod provider;
mod source;
mod support;
