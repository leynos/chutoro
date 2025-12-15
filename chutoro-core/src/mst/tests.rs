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

fn check_forest_invariants(node_count: usize, edges: &[MstEdge]) -> usize {
    let mut parent: Vec<usize> = (0..node_count).collect();

    fn find(parent: &mut [usize], node: usize) -> usize {
        let mut current = node;
        while parent[current] != current {
            let grandparent = parent[parent[current]];
            parent[current] = grandparent;
            current = parent[current];
        }
        current
    }

    fn union(parent: &mut [usize], left: usize, right: usize) -> bool {
        let left_root = find(parent, left);
        let right_root = find(parent, right);
        if left_root == right_root {
            return false;
        }
        parent[right_root] = left_root;
        true
    }

    for edge in edges {
        assert!(edge.source() < node_count);
        assert!(edge.target() < node_count);
        assert!(edge.source() < edge.target());
        assert!(edge.weight().is_finite());
        assert!(union(&mut parent, edge.source(), edge.target()));
    }

    let mut roots = (0..node_count)
        .map(|node| find(&mut parent, node))
        .collect::<Vec<_>>();
    roots.sort_unstable();
    roots.dedup();
    roots.len()
}

#[test]
fn rejects_empty_graph() {
    let result = parallel_kruskal(0, &EdgeHarvest::default());
    assert!(matches!(result, Err(MstError::EmptyGraph)));
}

#[test]
fn rejects_out_of_bounds_node_ids() {
    let edges = harvest(&[(0, 3, 1.0, 0)]);
    let result = parallel_kruskal(3, &edges);
    assert!(matches!(
        result,
        Err(MstError::InvalidNodeId {
            node: 3,
            node_count: 3
        })
    ));
}

#[test]
fn rejects_non_finite_weight() {
    let edges = harvest(&[(0, 1, f32::NAN, 0)]);
    let result = parallel_kruskal(2, &edges);
    assert!(matches!(
        result,
        Err(MstError::NonFiniteWeight { left: 0, right: 1 })
    ));
}

#[test]
fn ignores_self_edges() {
    let edges = harvest(&[(0, 0, 1.0, 0), (0, 1, 2.0, 1)]);
    let result = parallel_kruskal(2, &edges).expect("valid graph must succeed");
    assert_eq!(result.component_count(), 1);
    assert_eq!(result.edges().len(), 1);
    assert_eq!(result.edges()[0].source(), 0);
    assert_eq!(result.edges()[0].target(), 1);
}

fn connected_unique_weights_case() -> (usize, EdgeHarvest, Vec<MstEdge>) {
    let node_count = 4;
    let edges = harvest(&[
        (0, 1, 1.0, 0),
        (1, 2, 2.0, 1),
        (2, 3, 3.0, 2),
        (0, 2, 6.0, 3),
        (0, 3, 10.0, 4),
    ]);

    let expected = vec![
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
    ];

    (node_count, edges, expected)
}

#[rstest]
#[case::connected_unique_weights(connected_unique_weights_case())]
fn returns_expected_mst_on_unique_weights(#[case] case: (usize, EdgeHarvest, Vec<MstEdge>)) {
    let (node_count, edges, expected) = case;
    let result = parallel_kruskal(node_count, &edges).expect("MST must succeed");

    assert_eq!(result.component_count(), 1);
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
    assert_eq!(
        result.edges().len(),
        node_count.saturating_sub(component_count)
    );
}

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

    for _ in 0..25 {
        let result = parallel_kruskal(node_count, &edges).expect("graph must succeed");
        let component_count = check_forest_invariants(node_count, result.edges());
        assert_eq!(component_count, 1);
        assert_eq!(result.component_count(), 1);
        assert_eq!(result.edges().len(), node_count - 1);
        assert!(result.edges().iter().all(|edge| edge.weight() == 1.0));
    }
}
