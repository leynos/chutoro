//! Structural invariant checks for the CPU HNSW graph.
//!
//! The checkers are surfaced via [`CpuHnsw::invariants`] so property-based
//! tests can assert graph health after each operation without reimplementing
//! internal traversal logic.

use std::{collections::VecDeque, fmt};

use thiserror::Error;

use super::{CpuHnsw, graph::Graph, node::Node, params::HnswParams};

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
    ///
    /// # Examples
    /// ```rust,ignore
    /// # use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         Ok((self.0[i] - self.0[j]).abs())
    /// #     }
    /// # }
    /// let params = HnswParams::new(4, 8).expect("params");
    /// let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 2.0]), params)
    ///     .expect("build must succeed");
    /// index.invariants().check_all().expect("graph must be valid");
    /// ```
    pub fn check_all(&self) -> Result<(), HnswInvariantViolation> {
        self.check_many(HnswInvariant::all())
    }

    /// Runs a custom subset of invariants in the provided order.
    ///
    /// # Examples
    /// ```rust,ignore
    /// # use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswInvariant, HnswParams};
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         Ok((self.0[i] - self.0[j]).abs())
    /// #     }
    /// # }
    /// # let params = HnswParams::new(4, 8).unwrap();
    /// # let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 2.0]), params).unwrap();
    /// index
    ///     .invariants()
    ///     .check_many([HnswInvariant::Reachability])
    ///     .expect("invariant must hold");
    /// ```
    pub fn check_many(
        &self,
        invariants: impl IntoIterator<Item = HnswInvariant>,
    ) -> Result<(), HnswInvariantViolation> {
        for invariant in invariants {
            self.check(invariant)?;
        }
        Ok(())
    }

    /// Runs a single invariant.
    ///
    /// # Examples
    /// ```rust,ignore
    /// # use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswInvariant, HnswParams};
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         Ok((self.0[i] - self.0[j]).abs())
    /// #     }
    /// # }
    /// # let params = HnswParams::new(4, 8).unwrap();
    /// # let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 2.0]), params).unwrap();
    /// index
    ///     .invariants()
    ///     .check(HnswInvariant::DegreeBounds)
    ///     .expect("bounds satisfied");
    /// ```
    pub fn check(&self, invariant: HnswInvariant) -> Result<(), HnswInvariantViolation> {
        self.dispatch(invariant)
    }

    /// Runs the layer-consistency invariant directly.
    ///
    /// # Examples
    /// ```rust,ignore
    /// # use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         Ok((self.0[i] - self.0[j]).abs())
    /// #     }
    /// # }
    /// # let params = HnswParams::new(4, 8).unwrap();
    /// # let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 2.0]), params).unwrap();
    /// index.invariants().layer_consistency().expect("valid layers");
    /// ```
    pub fn layer_consistency(&self) -> Result<(), HnswInvariantViolation> {
        self.dispatch(HnswInvariant::LayerConsistency)
    }

    /// Runs the degree-bound invariant directly.
    ///
    /// # Examples
    /// ```rust,ignore
    /// # use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         Ok((self.0[i] - self.0[j]).abs())
    /// #     }
    /// # }
    /// # let params = HnswParams::new(4, 8).unwrap();
    /// # let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 2.0]), params).unwrap();
    /// index.invariants().degree_bounds().expect("degrees valid");
    /// ```
    pub fn degree_bounds(&self) -> Result<(), HnswInvariantViolation> {
        self.dispatch(HnswInvariant::DegreeBounds)
    }

    /// Runs the reachability invariant directly.
    ///
    /// # Examples
    /// ```rust,ignore
    /// # use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         Ok((self.0[i] - self.0[j]).abs())
    /// #     }
    /// # }
    /// # let params = HnswParams::new(4, 8).unwrap();
    /// # let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 2.0]), params).unwrap();
    /// index.invariants().reachability().expect("connected graph");
    /// ```
    pub fn reachability(&self) -> Result<(), HnswInvariantViolation> {
        self.dispatch(HnswInvariant::Reachability)
    }

    /// Runs the bidirectional-link invariant directly.
    ///
    /// # Examples
    /// ```rust,ignore
    /// # use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         Ok((self.0[i] - self.0[j]).abs())
    /// #     }
    /// # }
    /// # let params = HnswParams::new(4, 8).unwrap();
    /// # let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 2.0]), params).unwrap();
    /// index
    ///     .invariants()
    ///     .bidirectional_links()
    ///     .expect("links are symmetric");
    /// ```
    pub fn bidirectional_links(&self) -> Result<(), HnswInvariantViolation> {
        self.dispatch(HnswInvariant::BidirectionalLinks)
    }

    fn dispatch(&self, invariant: HnswInvariant) -> Result<(), HnswInvariantViolation> {
        self.with_graph(|graph, params| match invariant {
            HnswInvariant::LayerConsistency => check_layer_consistency(graph),
            HnswInvariant::DegreeBounds => check_degree_bounds(graph, params),
            HnswInvariant::Reachability => check_reachability(graph),
            HnswInvariant::BidirectionalLinks => check_bidirectional(graph),
        })
    }

    fn with_graph<R>(
        &self,
        f: impl FnOnce(&Graph, &HnswParams) -> Result<R, HnswInvariantViolation>,
    ) -> Result<R, HnswInvariantViolation> {
        let guard = self.index.graph.read().expect("graph lock poisoned");
        f(&guard, &self.index.params)
    }
}

fn check_layer_consistency(graph: &Graph) -> Result<(), HnswInvariantViolation> {
    let validator = LayerValidator::new(graph);
    for (source, node) in graph.nodes_iter() {
        for (level, target) in node.iter_neighbours() {
            validator.ensure(source, target, level)?;
        }
    }
    if let Some(entry) = graph.entry() {
        validator.ensure(entry.node, entry.node, entry.level)?;
    }
    Ok(())
}

fn check_degree_bounds(graph: &Graph, params: &HnswParams) -> Result<(), HnswInvariantViolation> {
    let upper = params.max_connections();
    let base_limit = upper.saturating_mul(2);
    for (node_id, node) in graph.nodes_iter() {
        for level in 0..node.level_count() {
            let limit = if level == 0 { base_limit } else { upper };
            let degree = node.neighbours(level).len();
            if degree > limit {
                return Err(HnswInvariantViolation::DegreeBounds {
                    node: node_id,
                    layer: level,
                    degree,
                    limit,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
struct BfsContext {
    visited: Vec<bool>,
    queue: VecDeque<usize>,
}

impl BfsContext {
    fn new(capacity: usize) -> Self {
        Self {
            visited: vec![false; capacity],
            queue: VecDeque::new(),
        }
    }
}

fn check_reachability(graph: &Graph) -> Result<(), HnswInvariantViolation> {
    if graph.nodes_iter().next().is_none() {
        return Ok(());
    }
    let entry = graph
        .entry()
        .ok_or(HnswInvariantViolation::MissingEntryPoint)?;
    let validator = LayerValidator::new(graph);
    validator.ensure(entry.node, entry.node, entry.level)?;

    let mut context = BfsContext::new(validator.capacity());
    context.visited[entry.node] = true;
    context.queue.push_back(entry.node);

    bfs_traverse(graph, &validator, &mut context)?;
    check_all_nodes_reachable(graph, &context.visited)
}

fn bfs_traverse(
    graph: &Graph,
    validator: &LayerValidator<'_>,
    context: &mut BfsContext,
) -> Result<(), HnswInvariantViolation> {
    while let Some(node_id) = context.queue.pop_front() {
        let node = graph
            .node(node_id)
            .ok_or(HnswInvariantViolation::LayerConsistency {
                origin: node_id,
                target: node_id,
                layer: 0,
                detail: LayerConsistencyDetail::MissingNode,
            })?;
        process_neighbours(node, validator, node_id, context)?;
    }

    Ok(())
}

fn process_neighbours(
    node: &Node,
    validator: &LayerValidator<'_>,
    origin: usize,
    context: &mut BfsContext,
) -> Result<(), HnswInvariantViolation> {
    for (level, target) in node.iter_neighbours() {
        validator.ensure(origin, target, level)?;
        if context.visited[target] {
            continue;
        }
        context.visited[target] = true;
        context.queue.push_back(target);
    }
    Ok(())
}

fn check_all_nodes_reachable(
    graph: &Graph,
    visited: &[bool],
) -> Result<(), HnswInvariantViolation> {
    for (node_id, _) in graph.nodes_iter() {
        if !visited[node_id] {
            return Err(HnswInvariantViolation::UnreachableNode { node: node_id });
        }
    }
    Ok(())
}

fn check_bidirectional(graph: &Graph) -> Result<(), HnswInvariantViolation> {
    let validator = LayerValidator::new(graph);
    for (source, node) in graph.nodes_iter() {
        for (level, target) in node.iter_neighbours() {
            let neighbour = validator.ensure(source, target, level)?;
            if neighbour.neighbours(level).contains(&source) {
                continue;
            }
            return Err(HnswInvariantViolation::MissingBacklink {
                origin: source,
                target,
                layer: level,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct LayerValidator<'a> {
    graph: &'a Graph,
    capacity: usize,
}

impl<'a> LayerValidator<'a> {
    fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            capacity: graph.capacity(),
        }
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure(
        &self,
        origin: usize,
        target: usize,
        layer: usize,
    ) -> Result<&'a Node, HnswInvariantViolation> {
        if target >= self.capacity {
            return Err(HnswInvariantViolation::LayerConsistency {
                origin,
                target,
                layer,
                detail: LayerConsistencyDetail::MissingNode,
            });
        }
        let node = self
            .graph
            .node(target)
            .ok_or(HnswInvariantViolation::LayerConsistency {
                origin,
                target,
                layer,
                detail: LayerConsistencyDetail::MissingNode,
            })?;
        if node.level_count() <= layer {
            return Err(HnswInvariantViolation::LayerConsistency {
                origin,
                target,
                layer,
                detail: LayerConsistencyDetail::MissingLayer {
                    available: node.level_count(),
                },
            });
        }
        Ok(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        datasource::DataSource,
        error::DataSourceError,
        hnsw::{
            CpuHnsw,
            graph::{Graph, NodeContext},
            params::HnswParams,
        },
    };
    use rstest::rstest;

    #[derive(Clone)]
    struct Dummy(Vec<f32>);

    impl Dummy {
        fn new(values: Vec<f32>) -> Self {
            Self(values)
        }
    }

    impl DataSource for Dummy {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn name(&self) -> &str {
            "dummy"
        }

        fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
            let a = self
                .0
                .get(i)
                .ok_or(DataSourceError::OutOfBounds { index: i })?;
            let b = self
                .0
                .get(j)
                .ok_or(DataSourceError::OutOfBounds { index: j })?;
            Ok((a - b).abs())
        }
    }

    #[test]
    fn check_all_succeeds_for_valid_index() {
        let params = HnswParams::new(4, 8).expect("params").with_rng_seed(7);
        let index =
            CpuHnsw::build(&Dummy::new(vec![0.0, 1.0, 2.0, 3.0]), params).expect("build hnsw");
        index
            .invariants()
            .check_all()
            .expect("valid graph must pass");
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

        let err = check_layer_consistency(&graph).expect_err("invariant must fail");
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
            let reverse = graph.node_mut(id).expect("reverse").neighbours_mut(level);
            reverse.push(0);
        }

        let node = graph.node_mut(0).expect("entry").neighbours_mut(level);
        node.clear();
        node.extend(1..=degree);

        let err = check_degree_bounds(&graph, &params).expect_err("must overflow");
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
    fn reachability_fails_for_disconnected_node() {
        let params = HnswParams::new(4, 8).expect("params");
        let mut graph = Graph::with_capacity(params, 3);
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
            .expect("attach neighbour");
        graph
            .attach_node(NodeContext {
                node: 2,
                level: 0,
                sequence: 2,
            })
            .expect("attach disconnected");

        graph.node_mut(0).expect("entry").neighbours_mut(0).push(1);
        graph.node_mut(1).expect("one").neighbours_mut(0).push(0);

        let err = check_reachability(&graph).expect_err("node 2 unreachable");
        assert_eq!(err, HnswInvariantViolation::UnreachableNode { node: 2 });
    }

    #[test]
    fn bidirectional_detects_missing_backlink() {
        let params = HnswParams::new(4, 8).expect("params").with_max_level(1);
        let mut graph = Graph::with_capacity(params, 3);
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
            .expect("attach node");
        graph.node_mut(0).expect("entry").neighbours_mut(0).push(1);

        let err = check_bidirectional(&graph).expect_err("reverse link missing");
        assert_eq!(
            err,
            HnswInvariantViolation::MissingBacklink {
                origin: 0,
                target: 1,
                layer: 0,
            }
        );
    }
}
