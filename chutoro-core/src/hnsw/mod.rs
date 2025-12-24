//! CPU implementation of the Hierarchical Navigable Small World (HNSW) graph.
//!
//! The implementation mirrors the concurrency model described in the design
//! documents: Rayon drives parallel insertion, every search acquires a shared
//! lock on the graph, and write access is limited to the mutation window when
//! inserting a node.

mod cpu;
mod distance_cache;
mod error;
mod graph;
mod helpers;
mod insert;
mod invariants;
mod node;
mod params;
mod search;
mod types;
mod validate;

pub use self::{
    cpu::CpuHnsw,
    distance_cache::DistanceCacheConfig,
    error::{HnswError, HnswErrorCode},
    invariants::{HnswInvariant, HnswInvariantChecker, HnswInvariantViolation},
    params::HnswParams,
    types::{CandidateEdge, EdgeHarvest, Neighbour},
};

#[cfg(test)]
mod tests;

#[cfg(kani)]
mod kani_proofs;
