//! Structural forest invariants for canonicalisation, acyclicity, and ordering.

use super::*;
use rstest::rstest;

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

    for window in result.edges().windows(2) {
        assert!(
            window[0].weight() <= window[1].weight(),
            "MST edges should be in non-decreasing weight order"
        );
    }
}
