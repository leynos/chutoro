//! Unit tests for candidate edge harvesting during HNSW construction.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    num::NonZeroUsize,
};

use rstest::{fixture, rstest};

use crate::DataSource;
use crate::hnsw::insert::extract_candidate_edges;
use crate::hnsw::types::{InsertionPlan, LayerPlan};
use crate::hnsw::{CandidateEdge, CpuHnsw, EdgeHarvest, HnswError, HnswParams, Neighbour};

use super::fixtures::DummySource;
use super::support::is_coverage_job;

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

#[derive(Clone, Debug)]
struct HarvestFixtureCase {
    data: Vec<f32>,
    max_connections: usize,
    ef_construction: usize,
    seed: Option<u64>,
}

impl HarvestFixtureCase {
    fn capacity(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone, Debug)]
struct InitialInsertCase {
    fixture: HarvestFixtureCase,
    first_node: usize,
}

#[derive(Clone, Debug)]
struct DuplicateInsertCase {
    fixture: HarvestFixtureCase,
    node: usize,
}

#[fixture]
fn dummy_source(
    #[default(HarvestFixtureCase {
        data: vec![0.0, 1.0, 2.0],
        max_connections: 2,
        ef_construction: 4,
        seed: Some(42),
    })]
    fixture: HarvestFixtureCase,
) -> DummySource {
    DummySource::new(fixture.data)
}

#[fixture]
fn hnsw_params(
    #[default(HarvestFixtureCase {
        data: vec![0.0, 1.0, 2.0],
        max_connections: 2,
        ef_construction: 4,
        seed: Some(42),
    })]
    fixture: HarvestFixtureCase,
) -> Result<HnswParams, Box<dyn Error>> {
    let params = HnswParams::new(fixture.max_connections, fixture.ef_construction)?;
    Ok(if let Some(rng_seed) = fixture.seed {
        params.with_rng_seed(rng_seed)
    } else {
        params
    })
}

#[fixture]
fn cpu_hnsw(
    #[default(HarvestFixtureCase {
        data: vec![0.0, 1.0, 2.0],
        max_connections: 2,
        ef_construction: 4,
        seed: Some(42),
    })]
    fixture: HarvestFixtureCase,
) -> Result<CpuHnsw, Box<dyn Error>> {
    Ok(CpuHnsw::with_capacity(
        hnsw_params(fixture.clone())?,
        fixture.capacity(),
    )?)
}

#[fixture]
fn comparison_cpu_hnsw(
    #[default(HarvestFixtureCase {
        data: vec![0.0, 1.0, 2.0],
        max_connections: 2,
        ef_construction: 4,
        seed: Some(42),
    })]
    fixture: HarvestFixtureCase,
) -> Result<CpuHnsw, Box<dyn Error>> {
    cpu_hnsw(fixture)
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
    // Half the midpoint is a symmetric 50% tolerance. Basing the allowance on
    // only the smaller sample made the accepted variance depend on build order.
    let base_tolerance = min_edges.saturating_add(max_edges).div_ceil(4).max(2);
    // Under coverage instrumentation, Rayon's thread scheduling is
    // significantly perturbed, causing insertion order to vary more
    // than in a standard build. The variance in harvested edge counts
    // scales with max_connections × ef_construction: ef_construction
    // bounds the candidate search width per insertion, and
    // max_connections bounds the neighbourhood retained.
    let coverage_tolerance = if is_coverage_job() {
        max_connections * ef_construction
    } else {
        0
    };
    let tolerance = base_tolerance + coverage_tolerance;

    assert!(
        max_edges <= min_edges + tolerance,
        "edge counts should be similar: {} vs {} (tolerance: {})",
        edges1.len(),
        edges2.len(),
        tolerance
    );
}

mod coverage;
