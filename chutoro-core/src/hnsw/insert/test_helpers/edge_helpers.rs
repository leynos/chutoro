//! Edge assertions and mutation helpers shared by HNSW tests and Kani proofs.

use crate::hnsw::graph::Graph;

pub(crate) fn add_edge_if_missing(graph: &mut Graph, origin: usize, target: usize, level: usize) {
    #[cfg(kani)]
    {
        let Some(node) = graph.node_mut(origin) else {
            kani::assert(false, "Kani origin node must exist");
            return;
        };
        let Some(neighbours) = node.neighbours_mut(level) else {
            kani::assert(false, "Kani origin must expose requested level");
            return;
        };
        if !neighbours.contains(&target) {
            neighbours.push(target);
        }
    }

    #[cfg(not(kani))]
    {
        let Some(node) = graph.node_mut(origin) else {
            debug_assert!(false, "missing origin node {origin}");
            return;
        };
        let Some(neighbours) = node.neighbours_mut(level) else {
            debug_assert!(false, "origin {origin} lacks requested level {level}");
            return;
        };
        if !neighbours.contains(&target) {
            neighbours.push(target);
        }
    }
}

pub(in crate::hnsw::insert) fn assert_no_edge(
    graph: &Graph,
    origin: usize,
    target: usize,
    level: usize,
) {
    if let Some(node) = graph.node(origin)
        && level < node.level_count()
    {
        assert!(
            !node.neighbours(level).contains(&target),
            "unexpected edge {origin}->{target} at level {level}",
        );
    }
}

/// Outcome of inspecting a node pair for a mutual edge at a given level.
///
/// Modelled as a query result so callers assert on a value rather than relying
/// on a helper to panic on their behalf.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EdgeSymmetry {
    /// Both nodes list one another at the requested level.
    Symmetric,
    /// The named node is absent from the graph.
    MissingNode(usize),
    /// The named node does not expose the requested level.
    LevelAbsent { node: usize, level_count: usize },
    /// The forward edge `origin -> target` is missing.
    MissingEdge { origin: usize, target: usize },
}

/// Classifies the edge relationship between `node_a` and `node_b` at `level`.
#[cfg(test)]
pub(crate) fn edge_symmetry(
    graph: &Graph,
    node_a: usize,
    node_b: usize,
    level: usize,
) -> EdgeSymmetry {
    let Some(a) = graph.node(node_a) else {
        return EdgeSymmetry::MissingNode(node_a);
    };
    let Some(b) = graph.node(node_b) else {
        return EdgeSymmetry::MissingNode(node_b);
    };

    if level >= a.level_count() {
        return EdgeSymmetry::LevelAbsent {
            node: node_a,
            level_count: a.level_count(),
        };
    }
    if level >= b.level_count() {
        return EdgeSymmetry::LevelAbsent {
            node: node_b,
            level_count: b.level_count(),
        };
    }

    if !a.neighbours(level).contains(&node_b) {
        return EdgeSymmetry::MissingEdge {
            origin: node_a,
            target: node_b,
        };
    }
    if !b.neighbours(level).contains(&node_a) {
        return EdgeSymmetry::MissingEdge {
            origin: node_b,
            target: node_a,
        };
    }
    EdgeSymmetry::Symmetric
}

/// Asserts that two nodes reference one another at `level`.
///
/// Implemented as a macro so a failure reports the calling test's line rather
/// than a shared helper's line.
#[cfg(test)]
macro_rules! assert_bidirectional_edge {
    ($graph:expr, $node_a:expr, $node_b:expr, $level:expr $(,)?) => {{
        let symmetry =
            $crate::hnsw::insert::test_helpers::edge_symmetry($graph, $node_a, $node_b, $level);
        assert_eq!(
            symmetry,
            $crate::hnsw::insert::test_helpers::EdgeSymmetry::Symmetric,
            "expected bidirectional edge {} <-> {} at level {}",
            $node_a,
            $node_b,
            $level,
        );
    }};
}

#[cfg(test)]
pub(crate) use assert_bidirectional_edge;
