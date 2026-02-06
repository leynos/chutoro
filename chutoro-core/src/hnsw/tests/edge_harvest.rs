//! Unit tests for candidate edge harvesting during HNSW construction.

use std::collections::{HashMap, HashSet};

use rstest::rstest;

use crate::hnsw::insert::extract_candidate_edges;
use crate::hnsw::types::{InsertionPlan, LayerPlan};
use crate::hnsw::{CandidateEdge, CpuHnsw, EdgeHarvest, HnswParams, Neighbour};

use super::fixtures::DummySource;

fn build_insertion_plan(layers: Vec<Vec<(usize, f32)>>) -> InsertionPlan {
    InsertionPlan {
        layers: layers
            .into_iter()
            .enumerate()
            .map(|(level, neighbours)| LayerPlan {
                level,
                neighbours: neighbours
                    .into_iter()
                    .map(|(id, distance)| Neighbour { id, distance })
                    .collect(),
            })
            .collect(),
    }
}

fn expected_edge_count(plan: &InsertionPlan, source_node: usize) -> usize {
    plan.layers
        .iter()
        .map(|layer| {
            layer
                .neighbours
                .iter()
                .filter(|neighbour| neighbour.id != source_node)
                .count()
        })
        .sum()
}

fn edge_key(edge: &CandidateEdge) -> (usize, usize, u32, u64) {
    (
        edge.source(),
        edge.target(),
        edge.distance().to_bits(),
        edge.sequence(),
    )
}

fn edge_multiset(edges: &[CandidateEdge]) -> HashMap<(usize, usize, u32, u64), usize> {
    let mut counts = HashMap::new();
    for edge in edges {
        *counts.entry(edge_key(edge)).or_insert(0) += 1;
    }
    counts
}

fn assert_edges_sorted_by_sequence_then_ord(edges: &[CandidateEdge]) {
    for window in edges.windows(2) {
        let (prev, curr) = (&window[0], &window[1]);
        let ordering = prev
            .sequence()
            .cmp(&curr.sequence())
            .then_with(|| prev.cmp(curr));
        assert!(
            ordering.is_le(),
            "edges must be sorted by (sequence, natural Ord): {prev:?} should come before {curr:?}",
        );
    }
}

#[derive(Debug)]
struct CanonicaliseCase {
    source: usize,
    target: usize,
    distance: f32,
    sequence: u64,
    expected_source: usize,
    expected_target: usize,
}

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

    // Validate all edge invariants in a single pass
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
        assert_ne!(
            edge.source(),
            edge.target(),
            "self-edge detected: {} -> {}",
            edge.source(),
            edge.target()
        );
        assert!(
            edge.distance().is_finite(),
            "non-finite distance: {}",
            edge.distance()
        );
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

    // Edge counts should be within reasonable tolerance (±50%)
    // since graph structure can vary with insertion order.
    // Coverage instrumentation can significantly affect Rayon's thread scheduling,
    // causing more variance in parallel insertion order and thus edge counts.
    let min_edges = edges1.len().min(edges2.len());
    let max_edges = edges1.len().max(edges2.len());
    let tolerance = (min_edges as f64 * 0.5).max(2.0) as usize;

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
            "node {node} should appear as an edge source",
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

    // Verify edges are sorted: primary key is sequence, secondary is natural Ord.
    //
    // Note: CandidateEdge::Ord uses distance as the primary key (not sequence),
    // so we must explicitly compare sequences first, then fall back to the
    // natural Ord only when sequences are equal.
    let edges_vec: Vec<CandidateEdge> = edges.iter().copied().collect();
    assert_edges_sorted_by_sequence_then_ord(&edges_vec);
}

#[rstest]
#[case::already_canonical(CanonicaliseCase {
    source: 1,
    target: 3,
    distance: 0.3,
    sequence: 20,
    expected_source: 1,
    expected_target: 3,
})]
#[case::reversed(CanonicaliseCase {
    source: 5,
    target: 2,
    distance: 0.5,
    sequence: 10,
    expected_source: 2,
    expected_target: 5,
})]
#[case::self_edge(CanonicaliseCase {
    source: 4,
    target: 4,
    distance: 0.7,
    sequence: 30,
    expected_source: 4,
    expected_target: 4,
})]
fn canonicalise_preserves_fields(#[case] case: CanonicaliseCase) {
    let edge = CandidateEdge::new(case.source, case.target, case.distance, case.sequence);
    let canonical = edge.canonicalise();

    assert_eq!(canonical.source(), case.expected_source);
    assert_eq!(canonical.target(), case.expected_target);
    assert!((canonical.distance() - case.distance).abs() < f32::EPSILON);
    assert_eq!(canonical.sequence(), case.sequence);
}

#[rstest]
fn candidate_edge_ordering_is_deterministic() {
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
#[case::empty_plan(7, 42, vec![], None)]
#[case::single_layer_no_self(0, 7, vec![vec![(1, 0.2), (2, 0.3)]], None)]
#[case::filters_self(3, 9, vec![vec![(3, 0.1), (4, 0.2)]], None)]
#[case::self_edges_only(4, 11, vec![vec![(4, 0.1), (4, 0.2)], vec![]], None)]
#[case::duplicate_target_across_layers(
    1,
    5,
    vec![vec![(2, 0.1), (1, 0.2)], vec![(2, 0.3)]],
    Some((2, 2)),
)]
fn extract_candidate_edges_invariants(
    #[case] source_node: usize,
    #[case] source_sequence: u64,
    #[case] layers: Vec<Vec<(usize, f32)>>,
    #[case] duplicate_expectation: Option<(usize, usize)>,
) {
    let plan = build_insertion_plan(layers);
    let edges = extract_candidate_edges(source_node, source_sequence, &plan);

    assert_eq!(edges.len(), expected_edge_count(&plan, source_node));
    for edge in &edges {
        assert_eq!(edge.source(), source_node);
        assert_ne!(edge.target(), source_node);
        assert_eq!(edge.sequence(), source_sequence);
    }

    if let Some((target, expected_count)) = duplicate_expectation {
        let observed = edges.iter().filter(|edge| edge.target() == target).count();
        assert_eq!(observed, expected_count);
    }
}

#[rstest]
#[case::unsorted(vec![
    CandidateEdge::new(2, 0, 0.8, 2),
    CandidateEdge::new(1, 0, 0.5, 1),
    CandidateEdge::new(1, 2, 0.5, 1),
    CandidateEdge::new(2, 0, 0.8, 2),
])]
#[case::already_sorted(vec![
    CandidateEdge::new(0, 1, 0.2, 1),
    CandidateEdge::new(0, 2, 0.2, 1),
    CandidateEdge::new(1, 2, 0.3, 2),
])]
#[case::empty(Vec::new())]
fn edge_harvest_from_unsorted_sorts_and_preserves_edges(#[case] edges: Vec<CandidateEdge>) {
    let original = edges.clone();
    let harvest = EdgeHarvest::from_unsorted(edges);
    let sorted = harvest.into_inner();

    assert_edges_sorted_by_sequence_then_ord(&sorted);
    assert_eq!(edge_multiset(&sorted), edge_multiset(&original));
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
