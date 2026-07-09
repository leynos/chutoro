//! MST tests for connected and disconnected forests.

use super::*;

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

// ============================================================================
// Forest Structural Invariant Tests
// ============================================================================

#[rstest]
#[case::connected_triangle(3, &[(0, 1, 1.0, 0), (1, 2, 1.0, 1), (0, 2, 1.0, 2)], 1)]
#[case::disconnected_pair(4, &[(0, 1, 1.0, 0), (2, 3, 1.0, 1)], 2)]
#[case::single_node(1, &[], 1)]
#[case::two_isolated_nodes(2, &[], 2)]
#[case::linear_chain(4, &[(0, 1, 1.0, 0), (1, 2, 2.0, 1), (2, 3, 3.0, 2)], 1)]
fn forest_has_correct_edge_count(
    #[case] node_count: usize,
    #[case] edges: &[(usize, usize, f32, u64)],
    #[case] expected_components: usize,
) {
    let edge_harvest = harvest(edges);
    let result = parallel_kruskal(node_count, &edge_harvest).expect("MST should succeed");

    assert_eq!(result.component_count(), expected_components);

    // Forest should have n - c edges
    let expected_edge_count = node_count.saturating_sub(expected_components);
    assert_eq!(
        result.edges().len(),
        expected_edge_count,
        "forest should have n - c edges"
    );
}

#[rstest]
#[case::triangle(3, &[(0, 1, 1.0, 0), (1, 2, 1.0, 1), (0, 2, 1.0, 2)])]
#[case::square_with_diagonal(4, &[(0, 1, 1.0, 0), (1, 2, 1.0, 1), (2, 3, 1.0, 2), (3, 0, 1.0, 3), (0, 2, 1.0, 4)])]
#[case::complete_4(4, &[(0, 1, 1.0, 0), (0, 2, 2.0, 1), (0, 3, 3.0, 2), (1, 2, 4.0, 3), (1, 3, 5.0, 4), (2, 3, 6.0, 5)])]
fn forest_is_acyclic(#[case] node_count: usize, #[case] edges: &[(usize, usize, f32, u64)]) {
    let edge_harvest = harvest(edges);
    let result = parallel_kruskal(node_count, &edge_harvest).expect("MST should succeed");

    // Verify acyclicity using union-find
    let mut parent: Vec<usize> = (0..node_count).collect();

    for edge in result.edges() {
        let root_s = union_find_root(&mut parent, edge.source());
        let root_t = union_find_root(&mut parent, edge.target());
        assert_ne!(
            root_s,
            root_t,
            "MST should be acyclic: edge ({}, {}) creates a cycle",
            edge.source(),
            edge.target()
        );
        parent[root_t] = root_s;
    }
}

#[rstest]
#[case::simple_edge(&[(0, 1, 1.0, 0)])]
#[case::reversed_edge(&[(1, 0, 1.0, 0)])]
#[case::mixed_directions(&[(0, 1, 1.0, 0), (2, 1, 2.0, 1)])]
#[case::self_loop_candidate(&[(0, 0, 1.0, 0), (0, 1, 2.0, 1)])]
fn edges_are_canonicalized(#[case] edges: &[(usize, usize, f32, u64)]) {
    let node_count = 3;
    let edge_harvest = harvest(edges);
    let result = parallel_kruskal(node_count, &edge_harvest).expect("MST should succeed");

    // All edges should be in canonical form: source < target
    // and no self-loops should be present
    for edge in result.edges() {
        assert_ne!(
            edge.source(),
            edge.target(),
            "forest must not contain self-loop edges"
        );
        assert!(
            edge.source() < edge.target(),
            "edge ({}, {}) is not canonical (source should be < target)",
            edge.source(),
            edge.target()
        );
    }
}

#[test]
fn weights_are_non_decreasing_in_mst() {
    let node_count = 5;
    let edges = harvest(&[
        (0, 1, 5.0, 0),
        (1, 2, 3.0, 1),
        (2, 3, 7.0, 2),
        (3, 4, 1.0, 3),
        (0, 4, 2.0, 4),
    ]);
    let result = parallel_kruskal(node_count, &edges).expect("MST should succeed");

    // MST edges should be sorted by weight (Kruskal property)
    for window in result.edges().windows(2) {
        assert!(
            window[0].weight() <= window[1].weight(),
            "MST edges should be in non-decreasing weight order"
        );
    }
}
