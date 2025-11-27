//! Connection limit utilities shared across insertion submodules.
//!
//! Provides a small helper to compute per-level neighbour limits used during
//! staging, reconciliation, and connectivity healing.

use crate::hnsw::params::connection_limit_for_level;

/// Computes the connection limit for a given level (doubled for level 0).
pub(super) fn compute_connection_limit(level: usize, max_connections: usize) -> usize {
    connection_limit_for_level(level, max_connections)
}
