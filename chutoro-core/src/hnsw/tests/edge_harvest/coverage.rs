//! Edge-harvest coverage and ordering tests.

use super::*;
use crate::hnsw::tests::support::is_coverage_job;

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
    // Coverage instrumentation inflates the cost of parallel HNSW builds;
    // use a smaller graph to stay well within the nextest timeout.
    let num_nodes = if is_coverage_job() { 8 } else { 20 };
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
        "both build methods should produce same index"
    );
}

#[rstest]
#[case(InitialInsertCase {
    fixture: HarvestFixtureCase {
        data: vec![0.0],
        max_connections: 2,
        ef_construction: 4,
        seed: Some(42),
    },
    first_node: 0,
})]
#[case(InitialInsertCase {
    fixture: HarvestFixtureCase {
        data: vec![1.0, 3.0, 5.0],
        max_connections: 2,
        ef_construction: 4,
        seed: Some(7),
    },
    first_node: 2,
})]
fn insert_harvesting_initial_insert_returns_empty_edges(
    #[case] case: InitialInsertCase,
    #[with(case.fixture.clone())] dummy_source: DummySource,
    #[with(case.fixture.clone())] cpu_hnsw: Result<CpuHnsw, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let cpu_hnsw = cpu_hnsw?;
    let edges = cpu_hnsw.insert_harvesting(case.first_node, &dummy_source)?;
    assert!(edges.is_empty(), "initial insert should return empty edges");
    Ok(())
}

#[rstest]
#[case(HarvestFixtureCase {
    data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
    max_connections: 2,
    ef_construction: 4,
    seed: Some(42),
})]
#[case(HarvestFixtureCase {
    data: vec![0.0, 0.5, 1.5, 3.0, 6.0, 10.0],
    max_connections: 3,
    ef_construction: 6,
    seed: Some(24),
})]
fn insert_harvesting_returns_valid_edges(
    #[case] _case: HarvestFixtureCase,
    #[with(_case.clone())] dummy_source: DummySource,
    #[with(_case.clone())] cpu_hnsw: Result<CpuHnsw, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let cpu_hnsw = cpu_hnsw?;
    cpu_hnsw.insert_harvesting(0, &dummy_source)?;

    for node in 1..dummy_source.len() {
        let edges = cpu_hnsw.insert_harvesting(node, &dummy_source)?;

        for edge in &edges {
            assert_eq!(
                edge.source(),
                node,
                "edge source should match inserted node"
            );
            assert!(
                edge.target() < dummy_source.len(),
                "target should be in bounds"
            );
            assert_ne!(edge.source(), edge.target(), "no self-edges allowed");
            assert!(edge.distance().is_finite(), "distance should be finite");
            assert!(edge.distance() >= 0.0, "distance should be non-negative");
        }
    }
    Ok(())
}

#[rstest]
#[case(DuplicateInsertCase {
    fixture: HarvestFixtureCase {
        data: vec![0.0, 1.0, 2.0],
        max_connections: 2,
        ef_construction: 4,
        seed: None,
    },
    node: 0,
})]
#[case(DuplicateInsertCase {
    fixture: HarvestFixtureCase {
        data: vec![0.0, 2.0, 4.0, 6.0],
        max_connections: 2,
        ef_construction: 4,
        seed: Some(42),
    },
    node: 2,
})]
fn insert_harvesting_duplicate_insert_is_rejected(
    #[case] case: DuplicateInsertCase,
    #[with(case.fixture.clone())] dummy_source: DummySource,
    #[with(case.fixture.clone())] cpu_hnsw: Result<CpuHnsw, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let cpu_hnsw = cpu_hnsw?;
    cpu_hnsw.insert_harvesting(case.node, &dummy_source)?;

    let err = cpu_hnsw
        .insert_harvesting(case.node, &dummy_source)
        .expect_err("duplicate insert fails");
    assert!(matches!(err, HnswError::DuplicateNode { node: duplicate } if duplicate == case.node));
    Ok(())
}

#[rstest]
#[case(HarvestFixtureCase {
    data: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    max_connections: 4,
    ef_construction: 16,
    seed: Some(42),
})]
fn insert_harvesting_matches_insert_graph_state(
    #[case] _case: HarvestFixtureCase,
    #[with(_case.clone())] dummy_source: DummySource,
    #[with(_case.clone())] cpu_hnsw: Result<CpuHnsw, Box<dyn Error>>,
    #[with(_case.clone())] comparison_cpu_hnsw: Result<CpuHnsw, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let ef = NonZeroUsize::new(dummy_source.len()).expect("source length must be non-zero");
    let index1 = cpu_hnsw?;
    let index2 = comparison_cpu_hnsw?;

    for node in 0..dummy_source.len() {
        index1.insert(node, &dummy_source)?;
        index2.insert_harvesting(node, &dummy_source)?;
    }

    assert_eq!(
        index1.len(),
        index2.len(),
        "indices should have same length"
    );

    for node in 0..dummy_source.len() {
        let insert_results = index1
            .search(&dummy_source, node, ef)
            .expect("search succeeds");
        let harvest_results = index2
            .search(&dummy_source, node, ef)
            .expect("search succeeds");

        assert_eq!(
            insert_results, harvest_results,
            "search results diverged for node {node}"
        );
    }
    Ok(())
}
