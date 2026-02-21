//! Benchmark support crate for chutoro.
//!
//! Provides synthetic data sources and parameter types used by Criterion
//! benchmarks for the four CPU pipeline stages: HNSW build, edge harvest,
//! MST computation, and hierarchy extraction.

pub mod ef_sweep;
pub mod error;
pub mod params;
pub mod profiling;
pub mod recall;
pub mod source;
