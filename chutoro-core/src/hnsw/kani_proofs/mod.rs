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
//! cargo kani -p chutoro-core --harness verify_bidirectional_links_commit_path_3_nodes
//! ```
//!
//! Or via the Makefile (practical harnesses):
//!
//! ```bash
//! make kani
//! ```
//!
//! Run the full suite (includes heavier 3-node harnesses):
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
mod eviction;
mod invariants;

use crate::hnsw::{graph::Graph, insert::test_helpers::add_edge_if_missing};

pub(super) struct EdgeAssertion {
    source: usize,
    target: usize,
    level: usize,
}

impl EdgeAssertion {
    pub(super) fn new(source: usize, target: usize, level: usize) -> Self {
        Self {
            source,
            target,
            level,
        }
    }
}

/// Returns `true` when source links to target at the given level.
pub(super) fn has_node_link(graph: &Graph, edge: EdgeAssertion) -> bool {
    graph
        .node(edge.source)
        .map(|n| n.neighbours(edge.level).contains(&edge.target))
        .unwrap_or(false)
}

pub(super) fn add_bidirectional_edge(
    graph: &mut Graph,
    origin: usize,
    target: usize,
    level: usize,
) {
    add_edge_if_missing(graph, origin, target, level);
    add_edge_if_missing(graph, target, origin, level);
}

pub(super) fn push_if_absent(list: &mut Vec<usize>, value: usize) {
    if !list.contains(&value) {
        list.push(value);
    }
}
