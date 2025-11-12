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
