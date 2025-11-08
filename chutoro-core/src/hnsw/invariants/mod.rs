//! Structural invariant checks for the CPU HNSW graph.
//!
//! The checkers are surfaced via [`CpuHnsw::invariants`] so property-based
//! tests can assert graph health after each operation without reimplementing
//! internal traversal logic.

mod bidirectional;
mod degree_bounds;
mod helpers;
mod layer_consistency;
mod reachability;

use std::fmt;

use thiserror::Error;

use crate::hnsw::{CpuHnsw, graph::Graph, params::HnswParams};

use self::{
    bidirectional::check_bidirectional, degree_bounds::check_degree_bounds,
    layer_consistency::check_layer_consistency, reachability::check_reachability,
};

/// Enumerates the structural invariants enforced by the CPU HNSW graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HnswInvariant {
    /// Ensures nodes referenced at a layer are initialised for that layer and
    /// all lower layers.
    LayerConsistency,
    /// Enforces the per-layer degree bounds described in the design docs.
    DegreeBounds,
    /// Guarantees every inserted node is reachable from the entry point.
    Reachability,
    /// Ensures edges are bidirectional at every layer.
    BidirectionalLinks,
}

impl HnswInvariant {
    /// Returns all invariants in the order they should be evaluated.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::hnsw::invariants::HnswInvariant;
    /// let invariants = HnswInvariant::all();
    /// assert_eq!(invariants.len(), 4);
    /// assert!(matches!(invariants[0], HnswInvariant::LayerConsistency));
    /// assert!(matches!(
    ///     invariants[3],
    ///     HnswInvariant::BidirectionalLinks
    /// ));
    /// ```
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [
            Self::LayerConsistency,
            Self::DegreeBounds,
            Self::Reachability,
            Self::BidirectionalLinks,
        ]
    }
}

/// Reason describing why a layer-consistency check failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerConsistencyDetail {
    /// The referenced node slot was never initialised.
    MissingNode,
    /// The referenced node exists but exposes fewer layers than required.
    MissingLayer { available: usize },
}

impl fmt::Display for LayerConsistencyDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingNode => f.write_str("target node is missing"),
            Self::MissingLayer { available } => {
                write!(f, "target node exposes only {available} level(s)")
            }
        }
    }
}

/// Reports an invariant violation surfaced by [`HnswInvariantChecker`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum HnswInvariantViolation {
    /// A node references a neighbour at a level the neighbour does not expose.
    #[error("node {origin} references {target} at layer {layer}, but {detail}")]
    LayerConsistency {
        /// Node that emitted the invalid reference.
        origin: usize,
        /// Referenced neighbour identifier.
        target: usize,
        /// Layer index of the reference.
        layer: usize,
        /// Detailed failure reason.
        detail: LayerConsistencyDetail,
    },
    /// A node exceeded the configured degree bound for a specific layer.
    #[error("node {node} has {degree} connection(s) on layer {layer}, exceeding limit {limit}")]
    DegreeBounds {
        /// Node whose adjacency exceeded the allowed bound.
        node: usize,
        /// Layer containing the overflow.
        layer: usize,
        /// Actual neighbouring nodes counted.
        degree: usize,
        /// Maximum permitted neighbours for the layer.
        limit: usize,
    },
    /// Configuration was invalid and prevented the invariant from running.
    #[error("invalid HNSW configuration: {message}")]
    ConfigError {
        /// Human-friendly explanation of the configuration issue.
        message: String,
    },
    /// Reported when a populated graph lacks an entry point.
    #[error("graph entry point missing despite populated nodes")]
    MissingEntryPoint,
    /// A node cannot be reached from the entry point using any layer.
    #[error("node {node} is unreachable from the entry point")]
    UnreachableNode {
        /// Identifier of the unreachable node.
        node: usize,
    },
    /// A directed edge is missing its counterpart at the same layer.
    #[error("edge {origin}->{target} at layer {layer} is missing the reverse link")]
    MissingBacklink {
        /// Source node that exposes the one-way edge.
        origin: usize,
        /// Target node lacking the reverse edge.
        target: usize,
        /// Layer index containing the asymmetric edge.
        layer: usize,
    },
}

/// Helper returned by [`CpuHnsw::invariants`] to run structural checks.
#[derive(Debug)]
pub struct HnswInvariantChecker<'index> {
    index: &'index CpuHnsw,
}

impl<'index> HnswInvariantChecker<'index> {
    pub(super) fn new(index: &'index CpuHnsw) -> Self {
        Self { index }
    }

    /// Runs all invariants, returning the first violation encountered.
    pub fn check_all(&self) -> Result<(), HnswInvariantViolation> {
        self.check_many(HnswInvariant::all())
    }

    /// Runs a custom subset of invariants in the provided order.
    pub fn check_many(
        &self,
        invariants: impl IntoIterator<Item = HnswInvariant>,
    ) -> Result<(), HnswInvariantViolation> {
        self.run_with_mode(invariants, EvaluationMode::FailFast)
    }

    /// Runs a single invariant.
    pub fn check(&self, invariant: HnswInvariant) -> Result<(), HnswInvariantViolation> {
        self.check_many([invariant])
    }

    /// Runs the layer-consistency invariant directly.
    pub fn layer_consistency(&self) -> Result<(), HnswInvariantViolation> {
        self.check(HnswInvariant::LayerConsistency)
    }

    /// Runs the degree-bound invariant directly.
    pub fn degree_bounds(&self) -> Result<(), HnswInvariantViolation> {
        self.check(HnswInvariant::DegreeBounds)
    }

    /// Runs the reachability invariant directly.
    pub fn reachability(&self) -> Result<(), HnswInvariantViolation> {
        self.check(HnswInvariant::Reachability)
    }

    /// Runs the bidirectional-link invariant directly.
    pub fn bidirectional_links(&self) -> Result<(), HnswInvariantViolation> {
        self.check(HnswInvariant::BidirectionalLinks)
    }

    /// Executes every invariant and returns the full set of violations.
    #[must_use]
    pub fn collect_all(&self) -> Vec<HnswInvariantViolation> {
        self.collect_many(HnswInvariant::all())
    }

    /// Executes the selected invariants and returns every violation discovered.
    #[must_use]
    pub fn collect_many(
        &self,
        invariants: impl IntoIterator<Item = HnswInvariant>,
    ) -> Vec<HnswInvariantViolation> {
        let mut violations = Vec::new();
        let _ = self.run_with_mode(invariants, EvaluationMode::Collect(&mut violations));
        violations
    }

    fn run_with_mode(
        &self,
        invariants: impl IntoIterator<Item = HnswInvariant>,
        mut mode: EvaluationMode<'_>,
    ) -> Result<(), HnswInvariantViolation> {
        self.with_context(|ctx| {
            for invariant in invariants {
                dispatch(ctx, invariant, &mut mode)?;
            }
            Ok(())
        })
    }

    fn with_context<R>(
        &self,
        f: impl FnOnce(GraphContext<'_>) -> Result<R, HnswInvariantViolation>,
    ) -> Result<R, HnswInvariantViolation> {
        let guard = self.index.graph.read().expect("graph lock poisoned");
        let ctx = GraphContext {
            graph: &guard,
            params: &self.index.params,
        };
        f(ctx)
    }
}

fn dispatch(
    ctx: GraphContext<'_>,
    invariant: HnswInvariant,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    match invariant {
        HnswInvariant::LayerConsistency => check_layer_consistency(ctx, mode),
        HnswInvariant::DegreeBounds => check_degree_bounds(ctx, mode),
        HnswInvariant::Reachability => check_reachability(ctx, mode),
        HnswInvariant::BidirectionalLinks => check_bidirectional(ctx, mode),
    }
}

#[derive(Clone, Copy)]
pub(super) struct GraphContext<'a> {
    graph: &'a Graph,
    params: &'a HnswParams,
}

pub(super) enum EvaluationMode<'a> {
    FailFast,
    Collect(&'a mut Vec<HnswInvariantViolation>),
}

impl EvaluationMode<'_> {
    fn record(&mut self, violation: HnswInvariantViolation) -> Result<(), HnswInvariantViolation> {
        match self {
            Self::FailFast => Err(violation),
            Self::Collect(sink) => {
                sink.push(violation);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests;
