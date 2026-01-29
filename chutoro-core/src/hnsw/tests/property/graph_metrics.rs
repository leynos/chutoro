//! Shared graph metrics and helper utilities for property-based tests.
//!
//! Centralises common computations (degrees, connectivity, RNN scores) so
//! multiple property suites can reuse the same definitions.

use std::collections::HashSet;

use crate::CandidateEdge;

use super::types::GraphMetadata;

/// Computes the degree of each node from an edge list.
///
/// Returns a vector where `degrees[i]` is the number of edges incident to node `i`.
/// For undirected graphs, each edge contributes 1 to both endpoints' degrees.
pub(super) fn compute_node_degrees(node_count: usize, edges: &[CandidateEdge]) -> Vec<usize> {
    let mut degrees = vec![0usize; node_count];
    for edge in edges {
        degrees[edge.source()] += 1;
        degrees[edge.target()] += 1;
    }
    degrees
}

/// Counts connected components using union-find with path compression.
///
/// Returns the number of distinct connected components in the graph.
pub(super) fn count_connected_components(node_count: usize, edges: &[CandidateEdge]) -> usize {
    if node_count == 0 {
        return 0;
    }

    let mut parent: Vec<usize> = (0..node_count).collect();

    fn find(parent: &mut [usize], mut node: usize) -> usize {
        let mut root = node;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[node] != root {
            let next = parent[node];
            parent[node] = root;
            node = next;
        }
        root
    }

    for edge in edges {
        let root_s = find(&mut parent, edge.source());
        let root_t = find(&mut parent, edge.target());
        if root_s != root_t {
            parent[root_t] = root_s;
        }
    }

    (0..node_count)
        .filter(|&i| find(&mut parent, i) == i)
        .count()
}

/// Computes the degree ceiling for the provided topology metadata.
pub(super) fn degree_ceiling_for_metadata(metadata: &GraphMetadata) -> usize {
    match metadata {
        GraphMetadata::Lattice { with_diagonals, .. } => {
            if *with_diagonals {
                8
            } else {
                4
            }
        }
        GraphMetadata::ScaleFree { node_count, .. } => node_count.saturating_sub(1),
        GraphMetadata::Random { node_count, .. } => node_count.saturating_sub(1),
        GraphMetadata::Disconnected {
            component_sizes, ..
        } => component_sizes
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .saturating_sub(1),
    }
}

/// Builds adjacency lists with distances for each node from an edge list.
fn build_adjacency_lists(node_count: usize, edges: &[CandidateEdge]) -> Vec<Vec<(usize, f32)>> {
    let mut adjacency: Vec<Vec<(usize, f32)>> = vec![Vec::new(); node_count];
    for edge in edges {
        adjacency[edge.source()].push((edge.target(), edge.distance()));
        adjacency[edge.target()].push((edge.source(), edge.distance()));
    }
    adjacency
}

/// Computes the top-k nearest neighbours for each node.
pub(super) fn top_k_neighbour_sets(
    node_count: usize,
    edges: &[CandidateEdge],
    k: usize,
) -> Vec<HashSet<usize>> {
    if k == 0 || node_count == 0 {
        return vec![HashSet::new(); node_count];
    }

    let adjacency = build_adjacency_lists(node_count, edges);
    adjacency
        .into_iter()
        .map(|mut neighbours| {
            neighbours.sort_by(|a, b| a.1.total_cmp(&b.1));
            neighbours.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

/// Counts symmetric relationships across all top-k neighbour sets.
fn count_symmetric_relationships(top_k_neighbours: &[HashSet<usize>]) -> (usize, usize) {
    let mut symmetric_count = 0usize;
    let mut total_relationships = 0usize;

    for (node, neighbours) in top_k_neighbours.iter().enumerate() {
        for &neighbour in neighbours {
            total_relationships += 1;
            if top_k_neighbours[neighbour].contains(&node) {
                symmetric_count += 1;
            }
        }
    }

    (symmetric_count, total_relationships)
}

/// Computes the RNN (Reverse Nearest Neighbour) score.
///
/// The score is the fraction of top-k neighbour relationships that are mutual.
pub(super) fn compute_rnn_score(node_count: usize, edges: &[CandidateEdge], k: usize) -> f64 {
    if k == 0 || node_count == 0 {
        return 1.0;
    }

    let top_k_neighbours = top_k_neighbour_sets(node_count, edges, k);
    let (symmetric_count, total_relationships) = count_symmetric_relationships(&top_k_neighbours);

    if total_relationships == 0 {
        1.0
    } else {
        symmetric_count as f64 / total_relationships as f64
    }
}

/// Computes the median of a slice of values.
pub(super) fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::GraphMetadata;
    use super::*;

    #[test]
    fn compute_node_degrees_empty_graph() {
        let degrees = compute_node_degrees(5, &[]);
        assert_eq!(degrees, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn compute_node_degrees_simple_chain() {
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(1, 2, 1.0, 1),
        ];
        let degrees = compute_node_degrees(3, &edges);
        assert_eq!(degrees, vec![1, 2, 1]);
    }

    #[test]
    fn count_connected_components_empty_graph() {
        assert_eq!(count_connected_components(0, &[]), 0);
    }

    #[test]
    fn count_connected_components_isolated_nodes() {
        assert_eq!(count_connected_components(5, &[]), 5);
    }

    #[test]
    fn count_connected_components_fully_connected() {
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(1, 2, 1.0, 1),
            CandidateEdge::new(0, 2, 1.0, 2),
        ];
        assert_eq!(count_connected_components(3, &edges), 1);
    }

    #[test]
    fn count_connected_components_two_components() {
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(2, 3, 1.0, 1),
        ];
        assert_eq!(count_connected_components(4, &edges), 2);
    }

    #[test]
    fn compute_rnn_score_empty_graph() {
        assert_eq!(compute_rnn_score(5, &[], 5), 1.0);
    }

    #[test]
    fn compute_rnn_score_k_zero_is_trivially_one() {
        let edges = vec![CandidateEdge::new(0, 1, 1.0, 0)];
        assert_eq!(compute_rnn_score(2, &edges, 0), 1.0);
    }

    #[test]
    fn compute_rnn_score_zero_nodes_is_trivially_one() {
        let edges: Vec<CandidateEdge> = Vec::new();
        assert_eq!(compute_rnn_score(0, &edges, 5), 1.0);
    }

    #[test]
    fn compute_rnn_score_symmetric_pair() {
        let edges = vec![CandidateEdge::new(0, 1, 1.0, 0)];
        assert_eq!(compute_rnn_score(2, &edges, 5), 1.0);
    }

    #[test]
    fn compute_rnn_score_asymmetric_star() {
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(0, 2, 2.0, 1),
            CandidateEdge::new(0, 3, 3.0, 2),
        ];
        let score = compute_rnn_score(4, &edges, 2);
        assert!((score - 0.8).abs() < 0.01);
    }

    #[test]
    fn median_even_count() {
        let mut values = vec![1.0, 3.0, 2.0, 4.0];
        assert!((median(&mut values) - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn median_odd_count() {
        let mut values = vec![3.0, 1.0, 2.0];
        assert!((median(&mut values) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn median_empty_slice_returns_zero() {
        let mut values: [f64; 0] = [];
        assert_eq!(median(&mut values), 0.0);
    }

    #[test]
    fn degree_ceiling_lattice_without_diagonals() {
        let metadata = GraphMetadata::Lattice {
            dimensions: (10, 10),
            with_diagonals: false,
        };
        assert_eq!(degree_ceiling_for_metadata(&metadata), 4);
    }

    #[test]
    fn degree_ceiling_lattice_with_diagonals() {
        let metadata = GraphMetadata::Lattice {
            dimensions: (10, 10),
            with_diagonals: true,
        };
        assert_eq!(degree_ceiling_for_metadata(&metadata), 8);
    }

    #[test]
    fn degree_ceiling_random_small_and_large() {
        let small = GraphMetadata::Random {
            node_count: 1,
            edge_probability: 0.1,
        };
        let large = GraphMetadata::Random {
            node_count: 12,
            edge_probability: 0.4,
        };
        assert_eq!(degree_ceiling_for_metadata(&small), 0);
        assert_eq!(degree_ceiling_for_metadata(&large), 11);
    }

    #[test]
    fn degree_ceiling_scale_free_small_and_large() {
        let small = GraphMetadata::ScaleFree {
            node_count: 1,
            edges_per_new_node: 1,
            exponent: 1.0,
        };
        let large = GraphMetadata::ScaleFree {
            node_count: 16,
            edges_per_new_node: 2,
            exponent: 1.4,
        };
        assert_eq!(degree_ceiling_for_metadata(&small), 0);
        assert_eq!(degree_ceiling_for_metadata(&large), 15);
    }

    #[test]
    fn degree_ceiling_disconnected_component_sizes() {
        let empty = GraphMetadata::Disconnected {
            component_count: 0,
            component_sizes: Vec::new(),
        };
        let singletons = GraphMetadata::Disconnected {
            component_count: 2,
            component_sizes: vec![1, 1],
        };
        let mixed = GraphMetadata::Disconnected {
            component_count: 3,
            component_sizes: vec![1, 3, 2],
        };
        assert_eq!(degree_ceiling_for_metadata(&empty), 0);
        assert_eq!(degree_ceiling_for_metadata(&singletons), 0);
        assert_eq!(degree_ceiling_for_metadata(&mixed), 2);
    }
}
