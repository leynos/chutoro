//! Distance primitives for built-in numeric metrics.
//!
//! The walking skeleton exposes scalar implementations for Euclidean and
//! cosine distances. These routines validate their inputs and surface detailed
//! errors so callers can react appropriately during ingestion or algorithmic
//! execution.

mod cosine;
mod euclidean;
mod helpers;
mod types;

pub use self::cosine::cosine_distance;
pub use self::euclidean::euclidean_distance;
pub use self::types::{CosineNorms, Distance, DistanceError, Norm, Result, VectorKind};
