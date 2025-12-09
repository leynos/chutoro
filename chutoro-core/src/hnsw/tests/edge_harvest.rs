//! Unit tests for candidate edge harvesting during HNSW construction.

use std::collections::HashSet;

use rstest::rstest;

use crate::hnsw::{CpuHnsw, HnswParams};

use super::fixtures::DummySource;

#[rstest]
#[case(3, 2, 4, 42)]
#[case(5, 2, 8, 123)]
#[case(10, 4, 16, 456)]
#[case(20, 8, 32, 789)]
fn build_with_edges_returns_valid_edges(
    #[case] num_nodes: usize,
    #[case] max_connections: usize,
    #[case] ef_construction: usize,
    #[case] seed: u64,
) {
    let data: Vec<f32> = (0..num_nodes).map(|i| i as f32).collect();
    let source = DummySource::new(data);
    let params = HnswParams::new(max_connections, ef_construction)
        .expect("params must be valid")
        .with_rng_seed(seed);

    let (index, edges) = CpuHnsw::build_with_edges(&source, params).expect("build must succeed");

    assert_eq!(index.len(), num_nodes);

    // All edges reference valid nodes
    for edge in &edges {
        assert!(
            edge.source() < num_nodes,
            "source {} out of bounds",
            edge.source()
        );
        assert!(
            edge.target() < num_nodes,
            "target {} out of bounds",
            edge.target()
        );
    }

    // No self-edges
    for edge in &edges {
        assert_ne!(
            edge.source(),
            edge.target(),
            "self-edge detected: {} -> {}",
            edge.source(),
            edge.target()
        );
    }

    // All distances are finite
    for edge in &edges {
        assert!(
            edge.distance().is_finite(),
            "non-finite distance: {}",
            edge.distance()
        );
    }

    // All distances are non-negative
    for edge in &edges {
        assert!(
            edge.distance() >= 0.0,
            "negative distance: {}",
            edge.distance()
        );
    }
}

#[rstest]
#[case(5, 2, 4, 42)]
#[case(10, 4, 8, 123)]
fn build_with_edges_has_consistent_count(
    #[case] num_nodes: usize,
    #[case] max_connections: usize,
    #[case] ef_construction: usize,
    #[case] seed: u64,
) {
    // Due to Rayon's non-deterministic thread scheduling, parallel insertions
    // can happen in different orders between runs. This affects the HNSW graph
    // structure and thus which candidate edges are discovered.
    //
    // We verify that multiple builds produce a similar number of edges (within
    // some tolerance), rather than requiring exact equality.
    let data: Vec<f32> = (0..num_nodes).map(|i| i as f32).collect();
    let source1 = DummySource::new(data.clone());
    let source2 = DummySource::new(data);

    let params1 = HnswParams::new(max_connections, ef_construction)
        .expect("params")
        .with_rng_seed(seed);
    let params2 = HnswParams::new(max_connections, ef_construction)
        .expect("params")
        .with_rng_seed(seed);

    let (index1, edges1) = CpuHnsw::build_with_edges(&source1, params1).expect("build 1");
    let (index2, edges2) = CpuHnsw::build_with_edges(&source2, params2).expect("build 2");

    // Both builds should produce the same number of nodes
    assert_eq!(index1.len(), index2.len(), "index sizes must match");

    // Edge counts should be within reasonable tolerance (±20%)
    // since graph structure can vary with insertion order
    let min_edges = edges1.len().min(edges2.len());
    let max_edges = edges1.len().max(edges2.len());
    let tolerance = (min_edges as f64 * 0.2).max(2.0) as usize;

    assert!(
        max_edges <= min_edges + tolerance,
        "edge counts should be similar: {} vs {} (tolerance: {})",
        edges1.len(),
        edges2.len(),
        tolerance
    );
}

#[rstest]
fn build_with_edges_covers_inserted_nodes() {
    let num_nodes = 10;
    let data: Vec<f32> = (0..num_nodes).map(|i| i as f32).collect();
    let source = DummySource::new(data);
    let params = HnswParams::new(4, 16).expect("params").with_rng_seed(42);

    let (_, edges) = CpuHnsw::build_with_edges(&source, params).expect("build");

    // Collect all nodes that appear as edge sources (these are the inserted nodes)
    let sources: HashSet<usize> = edges.iter().map(|e| e.source()).collect();

    // All nodes except the entry point (node 0) should appear as sources
    for node in 1..num_nodes {
        assert!(
            sources.contains(&node),
            "node {} should appear as an edge source",
            node
        );
    }
}

#[rstest]
fn build_with_edges_single_node_returns_empty_edges() {
    let source = DummySource::new(vec![0.0]);
    let params = HnswParams::new(2, 4).expect("params").with_rng_seed(42);

    let (index, edges) = CpuHnsw::build_with_edges(&source, params).expect("build");

    assert_eq!(index.len(), 1);
    assert!(
        edges.is_empty(),
        "single node build should produce no edges"
    );
}

#[rstest]
fn build_with_edges_two_nodes_produces_edges() {
    let source = DummySource::new(vec![0.0, 1.0]);
    let params = HnswParams::new(2, 4).expect("params").with_rng_seed(42);

    let (index, edges) = CpuHnsw::build_with_edges(&source, params).expect("build");

    assert_eq!(index.len(), 2);
    assert!(
        !edges.is_empty(),
        "two node build should produce at least one edge"
    );

    // The edge should connect node 1 to node 0 (node 1 discovers node 0 during insertion)
    assert!(
        edges.iter().any(|e| e.source() == 1 && e.target() == 0),
        "expected edge from node 1 to node 0"
    );
}

#[rstest]
fn build_with_edges_edges_sorted_by_sequence() {
    let num_nodes = 20;
    let data: Vec<f32> = (0..num_nodes).map(|i| i as f32).collect();
    let source = DummySource::new(data);
    let params = HnswParams::new(4, 16).expect("params").with_rng_seed(42);

    let (_, edges) = CpuHnsw::build_with_edges(&source, params).expect("build");

    // Verify edges are sorted by sequence (then by distance, source, target)
    for window in edges.windows(2) {
        let (prev, curr) = (&window[0], &window[1]);
        let ordering = prev
            .sequence()
            .cmp(&curr.sequence())
            .then_with(|| prev.cmp(curr));
        assert!(
            ordering.is_le(),
            "edges must be sorted by sequence: {:?} should come before {:?}",
            prev,
            curr
        );
    }
}

#[rstest]
fn canonicalise_normalises_edge_direction() {
    use crate::hnsw::CandidateEdge;

    let edge = CandidateEdge::new(5, 2, 0.5, 10);
    let canonical = edge.canonicalise();

    assert_eq!(canonical.source(), 2, "canonical source should be min");
    assert_eq!(canonical.target(), 5, "canonical target should be max");
    assert!((canonical.distance() - 0.5).abs() < f32::EPSILON);
    assert_eq!(canonical.sequence(), 10);

    // Already canonical edge should remain unchanged
    let already_canonical = CandidateEdge::new(1, 3, 0.3, 20);
    let result = already_canonical.canonicalise();
    assert_eq!(result.source(), 1);
    assert_eq!(result.target(), 3);
}

#[rstest]
fn candidate_edge_ordering_is_deterministic() {
    use crate::hnsw::CandidateEdge;

    let e1 = CandidateEdge::new(0, 1, 0.5, 10);
    let e2 = CandidateEdge::new(0, 2, 0.5, 10); // Same distance, different target
    let e3 = CandidateEdge::new(0, 1, 0.5, 11); // Same distance and target, different sequence

    // e1 < e2 (same distance, source; target 1 < 2)
    assert!(e1 < e2);

    // e1 < e3 (same distance, source, target; sequence 10 < 11)
    assert!(e1 < e3);

    // Distance takes priority
    let e4 = CandidateEdge::new(0, 1, 0.3, 100);
    assert!(e4 < e1, "lower distance should sort first");
}

#[rstest]
fn build_produces_same_index_as_build_with_edges() {
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let source = DummySource::new(data);
    let params = HnswParams::new(4, 16).expect("params").with_rng_seed(42);

    let index1 = CpuHnsw::build(&source, params.clone()).expect("build");
    let (index2, _) = CpuHnsw::build_with_edges(&source, params).expect("build_with_edges");

    assert_eq!(
        index1.len(),
        index2.len(),
        "both build methods should produce same size index"
    );
}
