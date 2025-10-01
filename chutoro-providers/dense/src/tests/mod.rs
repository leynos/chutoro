//! Dense provider test suite covering errors, ingestion, providers, sources, and shared fixtures.
pub(crate) use super::{DenseMatrixProvider, DenseMatrixProviderError, DenseSource};

mod errors;
mod ingest;
mod provider;
mod source;
mod support;
