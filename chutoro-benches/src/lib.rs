//! Benchmark support crate for chutoro.
//!
//! Provides synthetic data sources and parameter types used by Criterion
//! benchmarks for the four CPU pipeline stages: HNSW build, edge harvest,
//! MST computation, and hierarchy extraction.

pub mod error;
pub mod params;
pub mod profiling;
pub mod source;
