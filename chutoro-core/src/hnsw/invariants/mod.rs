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
mod tests {
    use super::{EvaluationMode, GraphContext, helpers, *};
    use crate::{
        datasource::DataSource,
        error::DataSourceError,
        hnsw::{
            graph::{Graph, NodeContext},
            params::HnswParams,
        },
    };
    use rstest::rstest;

    #[derive(Clone)]
    struct Dummy(Vec<f32>);

    impl DataSource for Dummy {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn name(&self) -> &str {
            "dummy"
        }
        fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
            Ok((self.0[i] - self.0[j]).abs())
        }
    }

    fn build_index() -> (CpuHnsw, Dummy) {
        let data = Dummy(vec![0.0, 1.0, 2.0, 3.0]);
        let params = HnswParams::new(4, 8).expect("params").with_rng_seed(7);
        let index = CpuHnsw::build(&data, params).expect("build hnsw");
        (index, data)
    }

    #[test]
    fn check_all_succeeds_for_valid_index() {
        let (index, _data) = build_index();
        index.invariants().check_all().expect("graph valid");
    }

    #[rstest]
    #[case::missing_node(|graph: &mut Graph| {
        graph.node_mut(0).expect("node 0").neighbours_mut(0).push(3);
    })]
    #[case::missing_layer(|graph: &mut Graph| {
        graph
            .node_mut(0)
            .expect("node 0")
            .neighbours_mut(1)
            .push(1);
    })]
    fn layer_consistency_reports_invalid_reference(#[case] mutate: fn(&mut Graph)) {
        let params = HnswParams::new(4, 8).expect("params").with_max_level(2);
        let mut graph = Graph::with_capacity(params.clone(), 4);
        graph
            .insert_first(NodeContext {
                node: 0,
                level: 1,
                sequence: 0,
            })
            .expect("insert entry");
        graph
            .attach_node(NodeContext {
                node: 1,
                level: 0,
                sequence: 1,
            })
            .expect("attach node");
        mutate(&mut graph);

        let validator = helpers::LayerValidator::new(&graph);
        let err = validator.ensure(0, 1, 1).expect_err("invariant must fail");
        matches!(err, HnswInvariantViolation::LayerConsistency { .. })
            .then_some(())
            .expect("layer consistency violation expected");
    }

    #[rstest]
    #[case(0, 9)]
    #[case(1, 5)]
    fn degree_bounds_detects_overflow(#[case] level: usize, #[case] degree: usize) {
        let params = HnswParams::new(4, 8).expect("params").with_max_level(2);
        let mut graph = Graph::with_capacity(params.clone(), 10);
        graph
            .insert_first(NodeContext {
                node: 0,
                level: level.max(1),
                sequence: 0,
            })
            .expect("insert entry");

        for id in 1..=degree {
            graph
                .attach_node(NodeContext {
                    node: id,
                    level,
                    sequence: id as u64,
                })
                .expect("attach neighbour");
            graph
                .node_mut(id)
                .expect("reverse")
                .neighbours_mut(level)
                .push(0);
        }

        let node = graph.node_mut(0).expect("entry").neighbours_mut(level);
        node.clear();
        node.extend(1..=degree);

        let ctx = GraphContext {
            graph: &graph,
            params: &params,
        };
        let mut mode = EvaluationMode::FailFast;
        let err = check_degree_bounds(ctx, &mut mode).expect_err("must overflow");
        if let HnswInvariantViolation::DegreeBounds {
            layer,
            degree: actual,
            ..
        } = err
        {
            assert_eq!(layer, level);
            assert_eq!(actual, degree);
        } else {
            panic!("expected degree bounds violation, got {err:?}");
        }
    }

    #[test]
    fn reachability_collects_all_unreachable_nodes() {
        let params = HnswParams::new(4, 8).expect("params");
        let mut graph = Graph::with_capacity(params.clone(), 4);
        graph
            .insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })
            .expect("insert entry");
        graph
            .attach_node(NodeContext {
                node: 1,
                level: 0,
                sequence: 1,
            })
            .expect("attach node 1");
        graph
            .attach_node(NodeContext {
                node: 2,
                level: 0,
                sequence: 2,
            })
            .expect("attach node 2");
        graph
            .attach_node(NodeContext {
                node: 3,
                level: 0,
                sequence: 3,
            })
            .expect("attach node 3");
        graph.node_mut(0).expect("entry").neighbours_mut(0).push(1);
        graph.node_mut(1).expect("one").neighbours_mut(0).push(0);

        let ctx = GraphContext {
            graph: &graph,
            params: &params,
        };
        let mut violations = Vec::new();
        let mut mode = EvaluationMode::Collect(&mut violations);
        check_reachability(ctx, &mut mode).expect("collect mode never errors");
        assert!(violations.iter().any(|violation| matches!(
            violation,
            HnswInvariantViolation::UnreachableNode { node: 2 }
        )));
        assert!(violations.iter().any(|violation| matches!(
            violation,
            HnswInvariantViolation::UnreachableNode { node: 3 }
        )));
    }

    #[test]
    fn collect_all_reports_multiple_violations() {
        let (index, _data) = build_index();
        {
            let mut graph = index.graph.write().expect("lock");
            clear_node(&mut graph, 0);
        }
        let violations = index.invariants().collect_all();
        assert!(
            violations.iter().any(|violation| matches!(
                violation,
                HnswInvariantViolation::UnreachableNode { .. }
            )),
            "collect_all should capture unreachable nodes"
        );
    }
    fn clear_node(graph: &mut Graph, node_id: usize) {
        if let Some(node) = graph.node_mut(node_id) {
            let levels = node.level_count();
            for level in 0..levels {
                node.neighbours_mut(level).clear();
            }
        }
    }
}
