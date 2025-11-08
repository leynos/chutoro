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

    while let Some(node_id) = context.queue.pop_front() {
        let node = ctx
            .graph
            .node(node_id)
            .ok_or(HnswInvariantViolation::LayerConsistency {
                origin: node_id,
                target: node_id,
                layer: 0,
                detail: super::LayerConsistencyDetail::MissingNode,
            })?;

        for (level, target) in node.iter_neighbours() {
            match validator.ensure(node_id, target, level) {
                Ok(_) if context.visited[target] => continue,
                Ok(_) => context.visit(target),
                Err(err) => mode.record(err)?,
            }
        }
    }

    for (node_id, _) in ctx.graph.nodes_iter() {
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
