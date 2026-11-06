//! Error taxonomy for minimum spanning tree (MST) construction.
//!
//! Kruskal's algorithm reports two distinct classes of failure: invalid input
//! (empty graphs, out-of-range node ids, non-finite weights) and runtime
//! failures inside the concurrent data structures (poisoned locks, violated
//! invariants). Keeping them in one place lets callers inspect, retry, or map
//! them without depending on the algorithm module itself.

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

impl MstError {
    /// Returns a stable, machine-readable error code for the variant.
    #[must_use]
    pub const fn code(&self) -> MstErrorCode {
        match self {
            Self::EmptyGraph => MstErrorCode::EmptyGraph,
            Self::InvalidNodeId { .. } => MstErrorCode::InvalidNodeId,
            Self::NonFiniteWeight { .. } => MstErrorCode::NonFiniteWeight,
            Self::LockPoisoned { .. } => MstErrorCode::LockPoisoned,
            Self::InvariantViolation { .. } => MstErrorCode::InvariantViolation,
        }
    }
}

/// Machine-readable error codes for [`MstError`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MstErrorCode {
    /// The caller requested an MST for an empty graph.
    EmptyGraph,
    /// An edge referenced a node id that is not present in the graph.
    InvalidNodeId,
    /// An edge contained a non-finite weight.
    NonFiniteWeight,
    /// A synchronization primitive became poisoned after a panic.
    LockPoisoned,
    /// An internal invariant was violated.
    InvariantViolation,
}

impl MstErrorCode {
    /// Returns the symbolic identifier for logging and metrics surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyGraph => "EMPTY_GRAPH",
            Self::InvalidNodeId => "INVALID_NODE_ID",
            Self::NonFiniteWeight => "NON_FINITE_WEIGHT",
            Self::LockPoisoned => "LOCK_POISONED",
            Self::InvariantViolation => "INVARIANT_VIOLATION",
        }
    }
}
