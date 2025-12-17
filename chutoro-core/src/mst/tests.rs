//! Unit tests for the parallel Kruskal MST implementation.

use rstest::rstest;

use crate::{CandidateEdge, EdgeHarvest};

use super::{MstEdge, MstError, parallel_kruskal};

fn harvest(edges: &[(usize, usize, f32, u64)]) -> EdgeHarvest {
    EdgeHarvest::new(
        edges
            .iter()
            .map(|(source, target, weight, sequence)| {
                CandidateEdge::new(*source, *target, *weight, *sequence)
            })
            .collect(),
    )
}

fn union_find_root(parent: &mut [usize], node: usize) -> usize {
    let mut current = node;
    while parent[current] != current {
        let grandparent = parent[parent[current]];
        parent[current] = grandparent;
        current = grandparent;
    }
    current
}

fn union_find_merge(parent: &mut [usize], left: usize, right: usize) -> bool {
    let left_root = union_find_root(parent, left);
    let right_root = union_find_root(parent, right);
    if left_root == right_root {
        return false;
    }
    parent[right_root] = left_root;
    true
}

fn check_forest_invariants(node_count: usize, edges: &[MstEdge]) -> usize {
    let mut parent: Vec<usize> = (0..node_count).collect();

    for edge in edges {
        assert!(edge.source() < node_count);
        assert!(edge.target() < node_count);
        assert!(edge.source() < edge.target());
        assert!(edge.weight().is_finite());
        assert!(union_find_merge(&mut parent, edge.source(), edge.target()));
    }

    let mut roots = (0..node_count)
        .map(|node| union_find_root(&mut parent, node))
        .collect::<Vec<_>>();
    roots.sort_unstable();
    roots.dedup();
    roots.len()
}

#[rstest]
#[case::empty_graph(0, &[], MstError::EmptyGraph)]
#[case::invalid_node_id(
    3,
    &[(0, 3, 1.0, 0)],
    MstError::InvalidNodeId {
        node: 3,
        node_count: 3,
    }
)]
#[case::nan_weight(
    2,
    &[(0, 1, f32::NAN, 0)],
    MstError::NonFiniteWeight { left: 0, right: 1 }
)]
#[case::pos_infinite_weight(
    2,
    &[(0, 1, f32::INFINITY, 0)],
    MstError::NonFiniteWeight { left: 0, right: 1 }
)]
#[case::neg_infinite_weight(
    2,
    &[(0, 1, f32::NEG_INFINITY, 0)],
    MstError::NonFiniteWeight { left: 0, right: 1 }
)]
fn rejects_invalid_inputs(
    #[case] node_count: usize,
    #[case] edges: &[(usize, usize, f32, u64)],
    #[case] expected_error: MstError,
) {
    let edge_harvest = harvest(edges);
    let result = parallel_kruskal(node_count, &edge_harvest);
    assert_eq!(
        result.expect_err("input should be rejected"),
        expected_error
    );
}

#[test]
fn ignores_self_edges() {
    let node_count = 2;
    let edges = harvest(&[(0, 0, 1.0, 0), (0, 1, 2.0, 1)]);
    let result = parallel_kruskal(node_count, &edges).expect("valid graph must succeed");
    assert_eq!(result.component_count(), 1);
    assert!(
        result.is_tree(),
        "MST of a connected graph should be a tree"
    );
    assert_eq!(
        result.edges().len(),
        node_count - 1,
        "MST of a connected graph should have N-1 edges"
    );
    assert_eq!(result.edges().len(), 1);
    assert_eq!(result.edges()[0].source(), 0);
    assert_eq!(result.edges()[0].target(), 1);
}

#[test]
fn returns_empty_forest_when_no_edges_are_usable() {
    let node_count = 2;
    let edges = harvest(&[(0, 0, 1.0, 0), (1, 1, 2.0, 1)]);
    let result = parallel_kruskal(node_count, &edges).expect("graph must be accepted");
    assert_eq!(result.component_count(), node_count);
    assert!(!result.is_tree());
    assert!(result.edges().is_empty());
}

#[test]
fn undirected_edges_are_canonicalised_and_deduplicated() {
    // Two nodes (0 and 1) with two parallel undirected edges between them:
    // (0, 1, w, s1) and (1, 0, w, s2). These should canonicalise to the same
    // undirected edge and be deduplicated down to a single MST edge.
    let node_count = 2;
    let edges = harvest(&[(0, 1, 1.0, 10), (1, 0, 1.0, 20)]);
    let result = parallel_kruskal(node_count, &edges).expect("graph must be accepted");

    assert_eq!(result.component_count(), 1);
    assert!(
        result.is_tree(),
        "MST of a connected graph should be a tree"
    );
    assert_eq!(
        result.edges().len(),
        node_count - 1,
        "MST of a connected graph should have N-1 edges"
    );

    let edge = &result.edges()[0];
    assert_eq!(edge.source(), 0);
    assert_eq!(edge.target(), 1);
    assert_eq!(edge.sequence(), 10);
}

/// Builds a test case for a connected graph with specified edges and expected MST.
fn build_connected_case(
    node_count: usize,
    input_edges: &[(usize, usize, f32, u64)],
    expected_edges: Vec<MstEdge>,
) -> (usize, EdgeHarvest, Vec<MstEdge>) {
    let edges = harvest(input_edges);
    (node_count, edges, expected_edges)
}

fn connected_unique_weights_case() -> (usize, EdgeHarvest, Vec<MstEdge>) {
    build_connected_case(
        4,
        &[
            (0, 1, 1.0, 0),
            (1, 2, 2.0, 1),
            (2, 3, 3.0, 2),
            (0, 2, 6.0, 3),
            (0, 3, 10.0, 4),
        ],
        vec![
            MstEdge {
                source: 0,
                target: 1,
                weight: 1.0,
                sequence: 0,
            },
            MstEdge {
                source: 1,
                target: 2,
                weight: 2.0,
                sequence: 1,
            },
            MstEdge {
                source: 2,
                target: 3,
                weight: 3.0,
                sequence: 2,
            },
        ],
    )
}

fn connected_duplicate_edges_case() -> (usize, EdgeHarvest, Vec<MstEdge>) {
    build_connected_case(
        4,
        &[
            // Same undirected edges with distinct sequences to exercise
            // deterministic tie-breaking during preparation.
            (1, 0, 1.0, 2),
            (0, 1, 1.0, 1),
            (2, 0, 1.0, 9),
            (0, 2, 1.0, 3),
            // Unique edge to connect node 3.
            (0, 3, 2.0, 4),
        ],
        vec![
            MstEdge {
                source: 0,
                target: 1,
                weight: 1.0,
                sequence: 1,
            },
            MstEdge {
                source: 0,
                target: 2,
                weight: 1.0,
                sequence: 3,
            },
            MstEdge {
                source: 0,
                target: 3,
                weight: 2.0,
                sequence: 4,
            },
        ],
    )
}

#[rstest]
#[case::connected_unique_weights(connected_unique_weights_case())]
#[case::connected_duplicate_edges(connected_duplicate_edges_case())]
fn returns_expected_mst_for_connected_graphs(#[case] case: (usize, EdgeHarvest, Vec<MstEdge>)) {
    let (node_count, edges, expected) = case;
    let result = parallel_kruskal(node_count, &edges).expect("MST must succeed");

    assert_eq!(result.component_count(), 1);
    assert!(
        result.is_tree(),
        "MST of a connected graph should be a tree"
    );
    assert_eq!(
        result.edges().len(),
        node_count - 1,
        "MST of a connected graph should have N-1 edges"
    );
    assert_eq!(result.edges(), expected.as_slice());
}

fn disconnected_case() -> (usize, EdgeHarvest) {
    let node_count = 5;
    let edges = harvest(&[(0, 1, 1.0, 0), (2, 3, 2.0, 1)]);
    (node_count, edges)
}

#[rstest]
#[case::disconnected(disconnected_case())]
fn returns_minimum_spanning_forest_for_disconnected_graph(#[case] case: (usize, EdgeHarvest)) {
    let (node_count, edges) = case;
    let result = parallel_kruskal(node_count, &edges).expect("forest must succeed");

    let component_count = check_forest_invariants(node_count, result.edges());
    assert_eq!(result.component_count(), component_count);
    assert!(!result.is_tree());
    assert_eq!(
        result.edges().len(),
        node_count.saturating_sub(component_count)
    );
}

/// Verify that the parallel Kruskal implementation produces deterministic,
/// cycle-free results when processing edges with equal weights.
///
/// The test runs the same graph multiple times and asserts that each run yields
/// the exact same ordered MST edge list (including the deterministic sequence
/// tie-breaks), guarding against race-induced non-determinism.
#[test]
fn handles_many_equal_weights_without_cycles() {
    let node_count = 6;
    let edges = harvest(&[
        (0, 1, 1.0, 0),
        (0, 2, 1.0, 1),
        (0, 3, 1.0, 2),
        (0, 4, 1.0, 3),
        (0, 5, 1.0, 4),
        (1, 2, 1.0, 5),
        (2, 3, 1.0, 6),
        (3, 4, 1.0, 7),
        (4, 5, 1.0, 8),
        (1, 5, 1.0, 9),
    ]);

    let mut expected_edges: Option<Vec<MstEdge>> = None;

    for _ in 0..25 {
        let result = parallel_kruskal(node_count, &edges).expect("graph must succeed");
        let actual_edges = result.edges().to_vec();
        let component_count = check_forest_invariants(node_count, result.edges());
        assert_eq!(component_count, 1);
        assert_eq!(result.component_count(), 1);
        assert!(result.is_tree());
        assert_eq!(result.edges().len(), node_count - 1);
        assert!(result.edges().iter().all(|edge| edge.weight() == 1.0));

        if let Some(expected) = expected_edges.as_ref() {
            assert_eq!(actual_edges.as_slice(), expected.as_slice());
        } else {
            expected_edges = Some(actual_edges);
        }
    }
}
