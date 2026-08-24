//! Kani formal verification harnesses for HNSW graph invariants.
//!
//! These harnesses use bounded model checking to exhaustively verify that
//! structural invariants hold for all possible graph configurations within
//! the specified bounds. Kani explores every possible combination of
//! nondeterministic choices, providing formal guarantees rather than
//! probabilistic coverage.
//!
//! # Running Harnesses
//!
//! ```bash
//! cargo kani -p chutoro-core --harness verify_bidirectional_links_smoke_2_nodes_1_layer
//! ```
//!
//! Or via the Makefile (practical harnesses):
//!
//! ```bash
//! make kani
//! ```
//!
//! Run the full suite (package-wide sweep):
//!
//! ```bash
//! make kani-full
//! ```
//!
//! # Relationship to Property Testing
//!
//! These harnesses complement the proptest-based property tests in
//! [`crate::hnsw::tests::property`]. While proptest provides probabilistic
//! coverage over large input spaces, Kani provides exhaustive coverage over
//! small, bounded configurations. Together they form a comprehensive
//! verification strategy.

mod bidirectional;
mod invariants;

use crate::hnsw::{graph::Graph, insert::test_helpers::add_edge_if_missing};

/// Seeds a forward and reverse edge pair between two existing nodes.
pub(super) fn add_bidirectional_edge(
    graph: &mut Graph,
    origin: usize,
    target: usize,
    level: usize,
) {
    add_edge_if_missing(graph, origin, target, level);
    add_edge_if_missing(graph, target, origin, level);
}
