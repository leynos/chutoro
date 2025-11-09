//! Reachability invariant enforcement for the CPU HNSW graph.
//!
//! Validates that every node remains reachable from the entry point across all
//! layers, reusing the shared `EvaluationMode` to either fail fast or collect
//! every missing edge and unreachable vertex.
use std::collections::VecDeque;

use super::{EvaluationMode, GraphContext, HnswInvariantViolation, helpers::LayerValidator};

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
        match traversal.validator.ensure(node_id, target, level) {
            Ok(_) if context.visited[target] => continue,
            Ok(_) => context.visit(target),
            Err(err) => mode.record(err)?,
        }
    }
    Ok(())
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
