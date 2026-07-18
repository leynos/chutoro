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

/// Builds an index with edge harvesting inside a dedicated single-threaded
/// Rayon pool, making insertion order (and therefore the harvested edges)
/// fully deterministic for a given seed.
fn build_with_edges_single_threaded(
    data: Vec<f32>,
    max_connections: usize,
    ef_construction: usize,
    seed: u64,
) -> (CpuHnsw, EdgeHarvest) {
    let source = DummySource::new(data);
    let params = HnswParams::new(max_connections, ef_construction)
        .expect("params")
        .with_rng_seed(seed);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-threaded Rayon pool");
    pool.install(|| CpuHnsw::build_with_edges(&source, params))
        .expect("build must succeed")
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
    // Rayon's thread scheduling makes parallel insertion order — and hence
    // the harvested edges — non-deterministic on the shared global pool, and
    // the variance grows with pool contention from concurrently running
    // tests. Comparing counts under a tolerance was therefore flaky under
    // plain `cargo test`, where every test shares one process (issue #155).
    //
    // Instead, run each build inside its own single-threaded Rayon pool:
    // insertions proceed in node order on one worker with a deterministic
    // per-worker RNG, so two same-seed builds must produce identical
    // harvests regardless of what else the process is running.
    let data: Vec<f32> = (0..num_nodes).map(|i| i as f32).collect();

    let (index1, edges1) =
        build_with_edges_single_threaded(data.clone(), max_connections, ef_construction, seed);
    let (index2, edges2) =
        build_with_edges_single_threaded(data, max_connections, ef_construction, seed);

    assert_eq!(index1.len(), index2.len(), "index sizes must match");
    assert_eq!(
        edges1.len(),
        edges2.len(),
        "same-seed single-threaded builds must harvest the same edge count",
    );
    assert_eq!(
        edges1, edges2,
        "same-seed single-threaded builds must harvest identical edges",
    );
}

mod coverage;
