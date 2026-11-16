//! Error taxonomy for minimum spanning tree (MST) construction.
//!
//! Kruskal's algorithm reports two distinct classes of failure: invalid input
//! (empty graphs, out-of-range node ids, non-finite weights) and runtime
//! failures inside the concurrent data structures (poisoned locks, violated
//! invariants). Keeping them in one place lets callers inspect, retry, or map
//! them without depending on the algorithm module itself.

use std::fmt;

use crate::error::define_error_codes;

/// Errors returned while computing a minimum spanning tree/forest.
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum MstError {
    /// The caller requested an MST for an empty graph.
    #[error("cannot compute an MST for an empty graph")]
    EmptyGraph,
    /// An edge referenced a node id that is not present in the graph.
    #[error("edge references node {node}, but node_count is {node_count}")]
    InvalidNodeId {
        /// The invalid node id referenced by an edge.
        node: usize,
        /// The number of nodes in the graph.
        node_count: usize,
    },
    /// An edge contained a non-finite weight.
    #[error("edge ({left}, {right}) has non-finite weight")]
    NonFiniteWeight {
        /// The left endpoint id (as provided).
        left: usize,
        /// The right endpoint id (as provided).
        right: usize,
    },
    /// A synchronization primitive became poisoned after a panic.
    #[error("lock for {resource} is poisoned")]
    LockPoisoned {
        /// Name of the locked resource that was poisoned.
        resource: &'static str,
    },
    /// An internal invariant was violated, indicating a logic error.
    #[error("MST invariant violated: {invariant} (index {index}, lock_count {lock_count})")]
    InvariantViolation {
        /// Name of the violated invariant to assist debugging.
        invariant: &'static str,
        /// The lock index that violated the invariant.
        index: usize,
        /// The number of locks available.
        lock_count: usize,
    },
}

define_error_codes! {
    /// Machine-readable error codes for [`MstError`].
    enum MstErrorCode for MstError {
        /// The caller requested an MST for an empty graph.
        EmptyGraph => EmptyGraph => "EMPTY_GRAPH",
        /// An edge referenced a node id that is not present in the graph.
        InvalidNodeId => InvalidNodeId { .. } => "INVALID_NODE_ID",
        /// An edge contained a non-finite weight.
        NonFiniteWeight => NonFiniteWeight { .. } => "NON_FINITE_WEIGHT",
        /// A synchronization primitive became poisoned after a panic.
        LockPoisoned => LockPoisoned { .. } => "LOCK_POISONED",
        /// An internal invariant was violated.
        InvariantViolation => InvariantViolation { .. } => "INVARIANT_VIOLATION",
    }
}
