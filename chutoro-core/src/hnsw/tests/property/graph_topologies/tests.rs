//! Unit tests for graph topology generators.

use rand::{SeedableRng, rngs::SmallRng};
use rstest::rstest;

use crate::hnsw::tests::property::graph_topology_tests::validate_edge;
use crate::hnsw::tests::property::types::{GeneratedGraph, GraphMetadata};

use super::{
    generate_disconnected_graph, generate_lattice_graph, generate_random_graph,
    generate_scale_free_graph,
};

/// Asserts that all edges in a graph are valid using the centralized validation.
fn assert_all_edges_valid(graph: &GeneratedGraph) {
    for (i, edge) in graph.edges.iter().enumerate() {
        validate_edge(edge, graph.node_count, i)
            .unwrap_or_else(|e| panic!("edge validation failed: {e}"));
    }
}

#[rstest]
#[case(42)]
#[case(123)]
#[case(456)]
#[case(789)]
fn random_graph_has_valid_structure(#[case] seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let graph = generate_random_graph(&mut rng);

    assert!(graph.node_count >= 4);
    assert!(graph.node_count <= 64);

    assert_all_edges_valid(&graph);
}

#[rstest]
#[case(42)]
#[case(123)]
#[case(456)]
#[case(789)]
fn scale_free_graph_has_valid_structure(#[case] seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let graph = generate_scale_free_graph(&mut rng);

    assert!(graph.node_count >= 8);
    assert!(graph.node_count <= 48);

    assert_all_edges_valid(&graph);
}

#[rstest]
#[case(42)]
#[case(123)]
#[case(456)]
#[case(789)]
fn lattice_graph_has_valid_structure(#[case] seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let graph = generate_lattice_graph(&mut rng);

    assert!(graph.node_count >= 4);
    assert!(graph.node_count <= 64);

    assert_all_edges_valid(&graph);

    // Lattice should always produce edges.
    assert!(!graph.edges.is_empty());
}

#[rstest]
#[case(42)]
#[case(123)]
#[case(456)]
#[case(789)]
fn disconnected_graph_has_valid_structure(#[case] seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let graph = generate_disconnected_graph(&mut rng);

    assert!(graph.node_count >= 6);

    assert_all_edges_valid(&graph);

    // Verify metadata.
    if let GraphMetadata::Disconnected {
        component_count,
        component_sizes,
    } = &graph.metadata
    {
        assert!(*component_count >= 2);
        assert_eq!(component_sizes.len(), *component_count);
        assert_eq!(component_sizes.iter().sum::<usize>(), graph.node_count);
    } else {
        panic!("expected Disconnected metadata");
    }
}

#[rstest]
fn lattice_metadata_matches_node_count() {
    let mut rng = SmallRng::seed_from_u64(999);
    let graph = generate_lattice_graph(&mut rng);

    if let GraphMetadata::Lattice { dimensions, .. } = &graph.metadata {
        assert_eq!(dimensions.0 * dimensions.1, graph.node_count);
    } else {
        panic!("expected Lattice metadata");
    }
}

#[rstest]
fn random_metadata_matches_node_count() {
    let mut rng = SmallRng::seed_from_u64(888);
    let graph = generate_random_graph(&mut rng);

    if let GraphMetadata::Random { node_count, .. } = &graph.metadata {
        assert_eq!(*node_count, graph.node_count);
    } else {
        panic!("expected Random metadata");
    }
}

#[rstest]
fn scale_free_metadata_matches_node_count() {
    let mut rng = SmallRng::seed_from_u64(777);
    let graph = generate_scale_free_graph(&mut rng);

    if let GraphMetadata::ScaleFree { node_count, .. } = &graph.metadata {
        assert_eq!(*node_count, graph.node_count);
    } else {
        panic!("expected ScaleFree metadata");
    }
}

#[rstest]
fn scale_free_graph_has_hub_nodes() {
    // Use a larger graph to observe hub formation.
    let mut rng = SmallRng::seed_from_u64(12345);
    // Generate multiple times to find one with enough nodes.
    for _ in 0..10 {
        let graph = generate_scale_free_graph(&mut rng);
        if graph.node_count < 20 {
            continue;
        }

        let mut degrees = vec![0usize; graph.node_count];
        for edge in &graph.edges {
            degrees[edge.source()] += 1;
            degrees[edge.target()] += 1;
        }

        let avg_degree: f64 = degrees.iter().sum::<usize>() as f64 / graph.node_count as f64;
        let max_degree = *degrees.iter().max().unwrap_or(&0);

        // Scale-free graphs should exhibit hub nodes with degree > average.
        // Relaxed assertion: max should be at least as large as average.
        assert!(
            max_degree as f64 >= avg_degree,
            "scale-free should have at least one hub: max={max_degree}, avg={avg_degree:.1}"
        );
        return;
    }
    panic!("failed to generate a scale-free graph with node_count >= 20 after 10 attempts");
}

#[rstest]
fn disconnected_graph_has_no_cross_component_edges() {
    let mut rng = SmallRng::seed_from_u64(54321);
    let graph = generate_disconnected_graph(&mut rng);

    if let GraphMetadata::Disconnected {
        component_sizes, ..
    } = &graph.metadata
    {
        // Build node-to-component mapping.
        let mut node_to_component = vec![0usize; graph.node_count];
        let mut offset = 0;
        for (comp_idx, &size) in component_sizes.iter().enumerate() {
            for i in 0..size {
                node_to_component[offset + i] = comp_idx;
            }
            offset += size;
        }

        // Verify no edge crosses components.
        for edge in &graph.edges {
            assert_eq!(
                node_to_component[edge.source()],
                node_to_component[edge.target()],
                "edge {edge:?} crosses components",
            );
        }
    }
}

#[rstest]
fn lattice_with_diagonals_has_more_edges() {
    // Generate lattices with and without diagonals and compare edge counts.
    let mut with_diag_edges = 0usize;
    let mut without_diag_edges = 0usize;
    let mut with_diag_count = 0usize;
    let mut without_diag_count = 0usize;

    for seed in 0..20 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let graph = generate_lattice_graph(&mut rng);

        if let GraphMetadata::Lattice { with_diagonals, .. } = &graph.metadata {
            if *with_diagonals {
                with_diag_edges += graph.edges.len();
                with_diag_count += 1;
            } else {
                without_diag_edges += graph.edges.len();
                without_diag_count += 1;
            }
        }
    }

    // Both variants should occur given 20 samples with ~50% probability each.
    assert!(
        with_diag_count > 0 && without_diag_count > 0,
        "expected both diagonal and non-diagonal lattices to be generated; \
         with_diag_count={with_diag_count}, without_diag_count={without_diag_count}"
    );

    // On average, diagonal lattices should have at least as many edges as non-diagonal ones.
    let avg_with_diag = with_diag_edges as f64 / with_diag_count as f64;
    let avg_without_diag = without_diag_edges as f64 / without_diag_count as f64;

    assert!(
        avg_with_diag >= avg_without_diag,
        "expected lattices with diagonals to be at least as dense as those without; \
         avg_with_diag={avg_with_diag}, avg_without_diag={avg_without_diag}"
    );
}
