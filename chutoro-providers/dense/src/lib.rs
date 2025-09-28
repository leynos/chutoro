//! Dense providers for f32 vectors backed by contiguous storage.

mod errors;
mod ingest;
mod provider;
mod source;

pub use errors::DenseMatrixProviderError;
pub use provider::DenseMatrixProvider;
pub use source::DenseSource;

#[cfg(test)]
mod tests;
