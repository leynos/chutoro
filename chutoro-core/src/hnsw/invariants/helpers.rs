use crate::hnsw::{graph::Graph, node::Node};

use super::{HnswInvariantViolation, LayerConsistencyDetail};

pub(super) fn for_each_edge(
    graph: &Graph,
    mut visitor: impl FnMut(usize, usize, usize) -> Result<(), HnswInvariantViolation>,
) -> Result<(), HnswInvariantViolation> {
    for (source, node) in graph.nodes_iter() {
        for (level, target) in node.iter_neighbours() {
            visitor(source, target, level)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LayerValidator<'a> {
    graph: &'a Graph,
    capacity: usize,
}

impl<'a> LayerValidator<'a> {
    pub(super) fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            capacity: graph.capacity(),
        }
    }

    pub(super) fn ensure(
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

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Checks if all edges in the graph are bidirectional.
///
/// Returns `true` if the invariant holds (every edge has a reverse edge at
/// the same layer), `false` otherwise. This simplified predicate is suitable
/// for use in Kani harnesses where a boolean result is preferred over
/// detailed violation reporting.
///
/// This function mirrors the logic in [`super::bidirectional::check_bidirectional`]
/// but returns a simple boolean rather than recording violations, ensuring
/// the Kani harness uses the same invariant definition as production code.
#[cfg(kani)]
pub(crate) fn is_bidirectional(graph: &Graph) -> bool {
    for (source, node) in graph.nodes_iter() {
        for (level, target) in node.iter_neighbours() {
            let Some(target_node) = graph.node(target) else {
                return false;
            };
            if target_node.level_count() <= level {
                return false;
            }
            if !target_node.neighbours(level).contains(&source) {
                return false;
            }
        }
    }
    true
}
