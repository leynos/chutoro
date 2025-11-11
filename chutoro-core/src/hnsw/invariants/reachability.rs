//! Reachability invariant enforcement for the CPU HNSW graph.
//!
//! Validates that every node remains reachable from the entry point across all
//! layers, reusing the shared `EvaluationMode` to either fail fast or collect
//! every missing edge and unreachable vertex.
use std::collections::VecDeque;

use super::{EvaluationMode, GraphContext, HnswInvariantViolation, helpers::LayerValidator};

/// Verifies that every node in the graph remains reachable from the declared
/// entry point by validating each layer and traversing the structure via BFS.
///
/// Records violations through `mode` so callers that aggregate failures can
/// continue collecting issues even when the function returns `Ok(())`.
///
/// # Parameters
/// - `ctx`: [`GraphContext`] supplying the HNSW graph and associated metadata
///   used by the layer validator and traversal logic.
/// - `mode`: [`EvaluationMode`] used to record or surface violations detected
///   during validation and traversal.
///
/// # Returns
/// [`Result::Ok`] when all nodes are reachable; otherwise returns
/// [`HnswInvariantViolation`] immediately if `mode` is fail-fast or after it
/// records the violation when `mode` is collecting.
///
/// # Errors
/// - Returns early with `Ok(())` when the graph is empty.
/// - Records `MissingEntryPoint` when the graph lacks an entry node.
/// - Propagates `LayerValidator::ensure` failures (e.g., broken layering) via
///   `mode` before returning.
/// - Surfaces violations encountered during `bfs_traverse`, such as invalid
///   neighbours discovered through `process_neighbour`.
/// - Emits unreachable-node violations from `check_all_nodes_visited` when BFS
///   fails to visit every node.
pub(super) fn check_reachability(
    ctx: GraphContext<'_>,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    if ctx.graph.nodes_iter().next().is_none() {
        return Ok(());
    }

    let entry = match ctx.graph.entry() {
        Some(entry) => entry,
        None => return mode.record(HnswInvariantViolation::MissingEntryPoint),
    };

    let validator = LayerValidator::new(ctx.graph);
    if let Err(err) = validator.ensure(entry.node, entry.node, entry.level) {
        return mode.record(err);
    }

    let mut context = BfsContext::new(validator.capacity());
    context.visit(entry.node);

    bfs_traverse(ctx.graph, &validator, &mut context, mode)?;
    check_all_nodes_visited(ctx.graph, &context, mode)
}

/// Performs the breadth-first traversal by draining `context.queue`, invoking
/// `process_single_node` for each dequeued vertex, and letting `mode` decide
/// whether violations should stop the walk. Accepts the HNSW `graph`, the
/// layer `validator`, the mutable BFS `context`, and the `mode` governing
/// violation handling, returning `Ok(())` when traversal finishes or
/// propagating any `Err(HnswInvariantViolation)` emitted by
/// `process_single_node`.
fn bfs_traverse(
    graph: &crate::hnsw::graph::Graph,
    validator: &LayerValidator<'_>,
    context: &mut BfsContext,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    let traversal = TraversalContext { graph, validator };
    while let Some(node_id) = context.queue.pop_front() {
        process_single_node(&traversal, node_id, context, mode)?;
    }
    Ok(())
}

fn process_single_node(
    traversal: &TraversalContext<'_>,
    node_id: usize,
    context: &mut BfsContext,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    let node = match traversal.graph.node(node_id) {
        Some(node) => node,
        None => {
            mode.record(HnswInvariantViolation::LayerConsistency {
                origin: node_id,
                target: node_id,
                layer: 0,
                detail: super::LayerConsistencyDetail::MissingNode,
            })?;
            return Ok(());
        }
    };

    for (level, target) in node.iter_neighbours() {
        let task = NeighbourTask {
            origin: node_id,
            level,
            target,
        };
        process_neighbour(traversal, task, context, mode)?;
    }
    Ok(())
}

#[inline(always)]
fn process_neighbour(
    traversal: &TraversalContext<'_>,
    task: NeighbourTask,
    context: &mut BfsContext,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    let NeighbourTask {
        origin,
        level,
        target,
    } = task;
    match traversal.validator.ensure(origin, target, level) {
        Ok(_) if context.visited[target] => {}
        Ok(_) => context.visit(target),
        Err(err) => mode.record(err)?,
    }
    Ok(())
}

struct NeighbourTask {
    origin: usize,
    level: usize,
    target: usize,
}

struct TraversalContext<'a> {
    graph: &'a crate::hnsw::graph::Graph,
    validator: &'a LayerValidator<'a>,
}

fn check_all_nodes_visited(
    graph: &crate::hnsw::graph::Graph,
    context: &BfsContext,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    for (node_id, _) in graph.nodes_iter() {
        if !context.visited[node_id] {
            mode.record(HnswInvariantViolation::UnreachableNode { node: node_id })?;
        }
    }
    Ok(())
}

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

    fn visit(&mut self, node: usize) {
        self.visited[node] = true;
        self.queue.push_back(node);
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the reachability helper functions and BFS context wiring.

    use super::*;
    use crate::hnsw::{
        graph::{Graph, NodeContext},
        params::HnswParams,
    };

    #[test]
    fn process_neighbour_visits_unseen_targets() {
        let graph = demo_graph();
        let validator = LayerValidator::new(&graph);
        let traversal = TraversalContext {
            graph: &graph,
            validator: &validator,
        };
        let mut context = BfsContext::new(validator.capacity());
        let mut mode = EvaluationMode::FailFast;
        let task = NeighbourTask {
            origin: 0,
            level: 0,
            target: 1,
        };

        process_neighbour(&traversal, task, &mut context, &mut mode)
            .expect("fresh targets should be enqueued");

        assert!(context.visited[1], "expected node 1 to be marked visited");
        assert_eq!(context.queue.pop_front(), Some(1));
    }

    #[test]
    fn process_neighbour_skips_already_visited_targets() {
        let graph = demo_graph();
        let validator = LayerValidator::new(&graph);
        let traversal = TraversalContext {
            graph: &graph,
            validator: &validator,
        };
        let mut context = BfsContext::new(validator.capacity());
        context.visit(1);
        context.queue.clear();
        let mut mode = EvaluationMode::FailFast;

        process_neighbour(
            &traversal,
            NeighbourTask {
                origin: 0,
                level: 0,
                target: 1,
            },
            &mut context,
            &mut mode,
        )
        .expect("visited nodes should be ignored");

        assert!(
            context.queue.is_empty(),
            "no duplicate visits should be queued"
        );
    }

    #[test]
    fn process_neighbour_records_layer_consistency_violations() {
        let graph = demo_graph();
        let validator = LayerValidator::new(&graph);
        let traversal = TraversalContext {
            graph: &graph,
            validator: &validator,
        };
        let mut context = BfsContext::new(validator.capacity());
        let mut violations = Vec::new();
        let mut mode = EvaluationMode::Collect(&mut violations);

        process_neighbour(
            &traversal,
            NeighbourTask {
                origin: 0,
                level: 0,
                target: 3,
            },
            &mut context,
            &mut mode,
        )
        .expect("collect mode should absorb violations");

        assert!(matches!(
            violations.as_slice(),
            [HnswInvariantViolation::LayerConsistency { target: 3, .. }]
        ));
    }

    fn demo_graph() -> Graph {
        let params = HnswParams::new(1, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 4);
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
        graph.node_mut(0).expect("node 0").neighbours_mut(0).push(1);
        graph.node_mut(1).expect("node 1").neighbours_mut(0).push(0);
        graph
    }
}
