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

/// Checks that no node has itself as a neighbour (no self-loops).
///
/// Returns `true` if the invariant holds (no node `u` has `u` in its neighbour
/// list at any layer), `false` otherwise. This simplified predicate is suitable
/// for use in Kani harnesses where a boolean result is preferred over detailed
/// violation reporting.
#[cfg(kani)]
pub(crate) fn has_no_self_loops(graph: &Graph) -> bool {
    for (node_id, node) in graph.nodes_iter() {
        for (_level, neighbour) in node.iter_neighbours() {
            if neighbour == node_id {
                return false;
            }
        }
    }
    true
}

/// Checks that all neighbour lists contain no duplicates.
///
/// Returns `true` if the invariant holds (every neighbour list at every layer
/// contains unique node identifiers), `false` otherwise. This simplified
/// predicate is suitable for use in Kani harnesses.
#[cfg(kani)]
pub(crate) fn has_unique_neighbours(graph: &Graph) -> bool {
    for (_node_id, node) in graph.nodes_iter() {
        for level in 0..node.level_count() {
            let neighbours = node.neighbours(level);
            for (i, &id) in neighbours.iter().enumerate() {
                if neighbours[i + 1..].contains(&id) {
                    return false;
                }
            }
        }
    }
    true
}

/// Checks entry-point validity and maximality.
///
/// Returns `true` if one of the following holds:
/// - The graph is empty and has no entry point.
/// - The graph is non-empty, the entry point exists, references a valid node,
///   and the entry level is at least as high as any other node's level.
///
/// This simplified predicate is suitable for use in Kani harnesses.
#[cfg(kani)]
pub(crate) fn is_entry_point_valid(graph: &Graph) -> bool {
    let has_nodes = graph.nodes_iter().next().is_some();

    if !has_nodes {
        return graph.entry().is_none();
    }

    let Some(entry) = graph.entry() else {
        return false;
    };

    let Some(entry_node) = graph.node(entry.node) else {
        return false;
    };

    // Entry level must be within the node's actual level count
    if entry.level >= entry_node.level_count() {
        return false;
    }

    // Entry level must be maximal across all nodes
    // (level_count() returns levels 0..level_count, so max level = level_count - 1)
    for (_id, node) in graph.nodes_iter() {
        // node.level_count() - 1 is the highest level for this node
        // entry.level must be >= that
        if node.level_count() > 0 && node.level_count() - 1 > entry.level {
            return false;
        }
    }

    true
}
