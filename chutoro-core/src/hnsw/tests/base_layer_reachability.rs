//! Test-only oracle for the normative base-layer reachability requirement.
//!
//! [`HnswInvariantChecker::reachability`] traverses neighbours at every layer,
//! so an upper-layer edge can bridge two otherwise separate level-0
//! components. The insertion policy in `docs/chutoro-design.md` §6.5.1 and
//! ADR 0010 requires reachability through level-0 edges alone. This module
//! supplies the independent traversal that tests use to check that stronger
//! requirement without changing the public invariant API.
//!
//! [`HnswInvariantChecker::reachability`]: crate::hnsw::HnswInvariantChecker::reachability

use std::collections::VecDeque;

use thiserror::Error;

use crate::hnsw::graph::Graph;

/// Reports why the base-layer oracle could not evaluate a graph.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum BaseLayerOracleError {
    /// A populated graph declares no entry point.
    #[error("graph has populated nodes but no entry point")]
    MissingEntryPoint,
    /// The entry identifier lies outside the graph's preallocated slots.
    #[error("entry node {entry} is outside graph capacity {capacity}")]
    EntryOutsideCapacity {
        /// Declared entry identifier.
        entry: usize,
        /// Number of preallocated node slots.
        capacity: usize,
    },
    /// The entry identifier names an unpopulated slot.
    #[error("entry node {entry} is not populated")]
    EntryNotPopulated {
        /// Declared entry identifier.
        entry: usize,
    },
    /// A level-0 neighbour reference is invalid for this graph.
    #[error("node {origin} has invalid base-layer neighbour {target}")]
    InvalidBaseEdge {
        /// Node exposing the neighbour reference.
        origin: usize,
        /// Invalid neighbour identifier.
        target: usize,
    },
}

/// Returns populated nodes unreachable from the entry point using level-0
/// edges only, in ascending identifier order.
///
/// An empty graph reports no unreachable nodes. Every populated graph, a
/// single node included, is validated through its entry point and then
/// traversed, so a graph whose only node has no declared entry still reports
/// [`BaseLayerOracleError::MissingEntryPoint`] instead of passing silently. The
/// traversal also validates every level-0 neighbour it follows, because a
/// malformed fixture would otherwise report a misleading reachability result.
///
/// # Errors
///
/// Returns [`BaseLayerOracleError`] when a populated graph has no entry point,
/// when the entry is outside the graph's slots or names an unpopulated slot,
/// or when a followed level-0 edge targets an invalid node.
///
/// # Examples
///
/// This helper is compiled only for tests, so the example is illustrative.
///
/// ```rust,ignore
/// let unreachable = unreachable_base_layer_nodes(&graph)?;
/// assert!(unreachable.is_empty());
/// ```
pub(crate) fn unreachable_base_layer_nodes(
    graph: &Graph,
) -> Result<Vec<usize>, BaseLayerOracleError> {
    let populated: Vec<usize> = graph.nodes_iter().map(|(id, _)| id).collect();
    if populated.is_empty() {
        return Ok(Vec::new());
    }

    let entry = graph
        .entry()
        .ok_or(BaseLayerOracleError::MissingEntryPoint)?;
    let capacity = graph.capacity();
    if entry.node >= capacity {
        return Err(BaseLayerOracleError::EntryOutsideCapacity {
            entry: entry.node,
            capacity,
        });
    }
    if graph.node(entry.node).is_none() {
        return Err(BaseLayerOracleError::EntryNotPopulated { entry: entry.node });
    }

    let reachable = traverse_base_layer(graph, entry.node, capacity)?;
    Ok(populated
        .into_iter()
        .filter(|id| !is_marked(&reachable, *id))
        .collect())
}

/// Marks every node reachable from `entry` through level-0 edges.
fn traverse_base_layer(
    graph: &Graph,
    entry: usize,
    capacity: usize,
) -> Result<Vec<bool>, BaseLayerOracleError> {
    let mut reachable = vec![false; capacity];
    let mut queue = VecDeque::from([entry]);
    mark_reachable(&mut reachable, entry);

    while let Some(current) = queue.pop_front() {
        let Some(node) = graph.node(current) else {
            continue;
        };
        for &neighbour in node.neighbours(0) {
            if graph.node(neighbour).is_none() {
                return Err(BaseLayerOracleError::InvalidBaseEdge {
                    origin: current,
                    target: neighbour,
                });
            }
            if !is_marked(&reachable, neighbour) {
                mark_reachable(&mut reachable, neighbour);
                queue.push_back(neighbour);
            }
        }
    }

    Ok(reachable)
}

/// Marks `node` reachable, ignoring an identifier outside the slot range.
fn mark_reachable(reachable: &mut [bool], node: usize) {
    if let Some(slot) = reachable.get_mut(node) {
        *slot = true;
    }
}

/// Reports whether `node` was previously marked reachable.
fn is_marked(reachable: &[bool], node: usize) -> bool {
    reachable.get(node).copied().unwrap_or(false)
}

#[cfg(test)]
mod tests {
    //! Self-tests for the base-layer oracle.

    use super::{BaseLayerOracleError, unreachable_base_layer_nodes};
    use crate::hnsw::{
        CpuHnsw, HnswParams,
        graph::{Graph, NodeContext},
        insert::test_helpers::add_edge_if_missing,
    };

    #[test]
    fn oracle_accepts_an_empty_graph() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let graph = Graph::with_capacity(params, 2);

        let unreachable =
            unreachable_base_layer_nodes(&graph).expect("an empty graph must be reachable");

        assert!(unreachable.is_empty());
    }

    #[test]
    fn oracle_rejects_a_populated_graph_without_an_entry_point() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 2);
        graph
            .attach_node(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })
            .expect("attach node without promoting the entry point");

        let error = unreachable_base_layer_nodes(&graph)
            .expect_err("a populated graph without an entry must be rejected");

        assert_eq!(error, BaseLayerOracleError::MissingEntryPoint);
    }

    #[test]
    fn oracle_accepts_a_single_populated_node_with_an_entry() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let mut graph = Graph::with_capacity(params, 2);
        graph
            .insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })
            .expect("insert the entry node");

        let unreachable =
            unreachable_base_layer_nodes(&graph).expect("a valid singleton must be reachable");

        assert!(unreachable.is_empty());
    }

    #[test]
    fn oracle_reports_a_node_reachable_only_through_an_upper_layer() {
        let params = HnswParams::new(2, 4).expect("params must be valid");
        let index = CpuHnsw::with_capacity(params, 2).expect("capacity must be valid");
        index
            .write_graph(|graph| {
                graph.insert_first(NodeContext {
                    node: 0,
                    level: 1,
                    sequence: 0,
                })?;
                graph.attach_node(NodeContext {
                    node: 1,
                    level: 1,
                    sequence: 1,
                })?;
                add_edge_if_missing(graph, 0, 1, 1);
                add_edge_if_missing(graph, 1, 0, 1);
                Ok(())
            })
            .expect("test graph must be writable");

        index
            .invariants()
            .reachability()
            .expect("cross-layer reachability must accept the upper-layer bridge");
        let unreachable = index
            .inspect_graph(unreachable_base_layer_nodes)
            .expect("graph inspection must succeed")
            .expect("oracle must evaluate a valid graph");
        assert_eq!(
            unreachable,
            vec![1],
            "node 1 is reachable only through the upper-layer bridge",
        );
    }
}
