//! Synthetic benchmark data sources.
//!
//! This module provides configurable generators for numeric and text
//! benchmarking datasets, together with a download-and-cache helper for
//! MNIST.

mod errors;
mod mnist;
mod numeric;
mod text;

pub use errors::SyntheticError;
pub use mnist::{MNIST_DIMENSIONS, MNIST_POINT_COUNT, MnistConfig};
pub use numeric::{
    Anisotropy, GaussianBlobConfig, ManifoldConfig, ManifoldPattern, SyntheticConfig,
    SyntheticSource,
};
pub use text::{SyntheticTextConfig, SyntheticTextSource};

#[cfg(test)]
mod tests;
