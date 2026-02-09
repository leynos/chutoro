//! Strategy builders for MST property-based tests.
//!
//! Provides graph generation strategies that produce varied weight
//! distributions and topologies designed to stress the parallel Kruskal
//! implementation. Each generator builds a list of [`CandidateEdge`]
//! instances with monotonic sequence numbers.

use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::CandidateEdge;

use super::types::{MstFixture, WeightDistribution};

/// Minimum node count for most generated graphs.
const MIN_NODES: usize = 8;
/// Maximum node count for most generated graphs.
const MAX_NODES: usize = 64;
/// Maximum node count for dense graphs (kept smaller to avoid quadratic
/// edge explosion).
const DENSE_MAX_NODES: usize = 32;

/// Generates MST fixtures covering all five weight distributions.
///
/// Uses `prop_oneof!` with weighting that biases towards the
/// `ManyIdentical` distribution (the most important stress case for
/// parallel tie-breaking).
pub(super) fn mst_fixture_strategy() -> impl Strategy<Value = MstFixture> {
    (any::<WeightDistribution>(), any::<u64>()).prop_map(|(distribution, seed)| {
        let mut rng = SmallRng::seed_from_u64(seed);
        generate_fixture(distribution, &mut rng)
    })
}

/// Generates a fixture for a specific weight distribution.
///
/// Useful for targeted rstest cases where the distribution is chosen
/// explicitly rather than sampled by proptest.
pub(super) fn generate_fixture(distribution: WeightDistribution, rng: &mut SmallRng) -> MstFixture {
    match distribution {
        WeightDistribution::Unique => generate_unique_weights(rng),
        WeightDistribution::ManyIdentical => generate_identical_weights(rng),
        WeightDistribution::Sparse => generate_sparse(rng),
        WeightDistribution::Dense => generate_dense(rng),
        WeightDistribution::Disconnected => generate_disconnected(rng),
    }
}

// ── Probabilistic graph helper ──────────────────────────────────────────

/// Configuration for probabilistic graph generation, grouping the
/// parameters that vary between weight-distribution strategies.
struct ProbabilisticGraphConfig {
    /// Upper bound for the random node count (inclusive).
    max_nodes: usize,
    /// Inclusive range from which the per-pair edge probability is sampled.
    edge_prob_range: (f64, f64),
    /// Weight distribution label for the resulting fixture.
    distribution: WeightDistribution,
}

/// Generates a graph by probabilistically adding edges between all unique
/// node pairs, using a caller-supplied weight generator.
///
/// Encapsulates the common pattern shared by `generate_unique_weights`,
/// `generate_identical_weights`, and `generate_dense`.
fn generate_probabilistic_graph(
    rng: &mut SmallRng,
    config: ProbabilisticGraphConfig,
    mut weight_generator: impl FnMut(&mut SmallRng) -> f32,
) -> MstFixture {
    let node_count = rng.gen_range(MIN_NODES..=config.max_nodes);
    let edge_probability: f64 = rng.gen_range(config.edge_prob_range.0..=config.edge_prob_range.1);
    let mut edges = Vec::new();
    let mut seq = 0u64;

    for i in 0..node_count {
        for j in (i + 1)..node_count {
            if rng.gen_bool(edge_probability) {
                let weight = weight_generator(rng);
                edges.push(CandidateEdge::new(i, j, weight, seq));
                seq += 1;
            }
        }
    }

    ensure_at_least_one_edge(node_count, &mut edges, &mut seq, rng);

    MstFixture {
        node_count,
        edges,
        distribution: config.distribution,
    }
}

/// Generates a probabilistic graph with continuous weights drawn from the
/// range \[0.1, 100.0).
fn generate_continuous_weight_graph(
    rng: &mut SmallRng,
    max_nodes: usize,
    edge_prob_range: (f64, f64),
    distribution: WeightDistribution,
) -> MstFixture {
    generate_probabilistic_graph(
        rng,
        ProbabilisticGraphConfig {
            max_nodes,
            edge_prob_range,
            distribution,
        },
        |r| r.gen_range(0.1_f32..100.0),
    )
}

// ── Unique weights ──────────────────────────────────────────────────────

/// Generates a graph where each edge has a distinct weight drawn from a
/// continuous range. This is the baseline correctness case where the MST
/// is unique (up to floating-point coincidence).
fn generate_unique_weights(rng: &mut SmallRng) -> MstFixture {
    generate_continuous_weight_graph(rng, MAX_NODES, (0.2, 0.6), WeightDistribution::Unique)
}

// ── Many identical weights ──────────────────────────────────────────────

/// Generates a graph where large groups of edges share the same weight.
///
/// This is the most important stress case — it creates contention in the
/// parallel tie-breaking logic and exercises deterministic weight-group
/// processing.
fn generate_identical_weights(rng: &mut SmallRng) -> MstFixture {
    let weight_pool_size = rng.gen_range(1..=3);
    let weight_pool: Vec<f32> = (0..weight_pool_size)
        .map(|_| rng.gen_range(1_u8..=10) as f32)
        .collect();

    generate_probabilistic_graph(
        rng,
        ProbabilisticGraphConfig {
            max_nodes: MAX_NODES,
            edge_prob_range: (0.3, 0.7),
            distribution: WeightDistribution::ManyIdentical,
        },
        move |r| weight_pool[r.gen_range(0..weight_pool.len())],
    )
}

// ── Sparse ──────────────────────────────────────────────────────────────

/// Generates a sparse graph by first building a random spanning tree
/// (guaranteeing connectivity) and then adding a small number of extra
/// edges.
fn generate_sparse(rng: &mut SmallRng) -> MstFixture {
    let node_count = rng.gen_range(MIN_NODES..=MAX_NODES);
    let mut edges = Vec::new();
    let mut seq = 0u64;

    // Build a random spanning tree via random permutation walk.
    let mut perm: Vec<usize> = (0..node_count).collect();
    shuffle(&mut perm, rng);
    for i in 1..node_count {
        let weight = rng.gen_range(0.1_f32..100.0);
        let (s, t) = canonical(perm[i - 1], perm[i]);
        edges.push(CandidateEdge::new(s, t, weight, seq));
        seq += 1;
    }

    // Add a small number of extra edges (roughly 0.5n to n).
    let extra_count = rng.gen_range(node_count / 2..=node_count);
    for _ in 0..extra_count {
        let i = rng.gen_range(0..node_count);
        let j = rng.gen_range(0..node_count);
        if i != j {
            let weight = rng.gen_range(0.1_f32..100.0);
            let (s, t) = canonical(i, j);
            edges.push(CandidateEdge::new(s, t, weight, seq));
            seq += 1;
        }
    }

    MstFixture {
        node_count,
        edges,
        distribution: WeightDistribution::Sparse,
    }
}

// ── Dense ───────────────────────────────────────────────────────────────

/// Generates a dense graph approaching a complete graph, with node count
/// capped at [`DENSE_MAX_NODES`] to avoid quadratic edge explosion.
fn generate_dense(rng: &mut SmallRng) -> MstFixture {
    generate_continuous_weight_graph(rng, DENSE_MAX_NODES, (0.7, 0.95), WeightDistribution::Dense)
}

// ── Disconnected ────────────────────────────────────────────────────────

/// Generates a graph with 2-5 disconnected components, each having random
/// internal structure. No cross-component edges are created.
fn generate_disconnected(rng: &mut SmallRng) -> MstFixture {
    let component_count = rng.gen_range(2..=5);
    let component_sizes: Vec<usize> = (0..component_count)
        .map(|_| rng.gen_range(3..=12))
        .collect();
    let node_count: usize = component_sizes.iter().sum();
    let mut builder = EdgeBuilder::default();
    let mut node_offset = 0;

    for &size in &component_sizes {
        builder.generate_component(node_offset, size, rng);
        node_offset += size;
    }

    MstFixture {
        node_count,
        edges: builder.edges,
        distribution: WeightDistribution::Disconnected,
    }
}

/// Accumulates candidate edges with monotonic sequence numbering.
#[derive(Default)]
struct EdgeBuilder {
    edges: Vec<CandidateEdge>,
    seq: u64,
}

impl EdgeBuilder {
    /// Adds a single candidate edge and advances the sequence counter.
    fn push(&mut self, source: usize, target: usize, weight: f32) {
        self.edges
            .push(CandidateEdge::new(source, target, weight, self.seq));
        self.seq += 1;
    }

    /// Generates edges for a single connected component within a
    /// disconnected graph, guaranteeing at least one edge when the
    /// component has two or more nodes.
    fn generate_component(&mut self, node_offset: usize, size: usize, rng: &mut SmallRng) {
        let edge_probability: f64 = rng.gen_range(0.3..=0.8);
        let start_len = self.edges.len();

        let pairs = all_pairs(node_offset, size);
        for (s, t) in pairs {
            if rng.gen_bool(edge_probability) {
                let weight = rng.gen_range(0.1_f32..100.0);
                self.push(s, t, weight);
            }
        }

        // Guarantee at least one edge per component (except singletons).
        if size >= 2 && self.edges.len() == start_len {
            let weight = rng.gen_range(0.1_f32..100.0);
            self.push(node_offset, node_offset + 1, weight);
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Returns the pair in canonical order `(min, max)`.
fn canonical(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Returns all unique undirected pairs `(node_offset + i, node_offset + j)`
/// where `i < j < size`.
fn all_pairs(node_offset: usize, size: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..size {
        for j in (i + 1)..size {
            pairs.push((node_offset + i, node_offset + j));
        }
    }
    pairs
}

/// Ensures the edge list contains at least one edge by inserting a
/// fallback edge between nodes 0 and 1.
fn ensure_at_least_one_edge(
    node_count: usize,
    edges: &mut Vec<CandidateEdge>,
    seq: &mut u64,
    rng: &mut SmallRng,
) {
    if edges.is_empty() && node_count >= 2 {
        let weight = rng.gen_range(0.1_f32..100.0);
        edges.push(CandidateEdge::new(0, 1, weight, *seq));
        *seq += 1;
    }
}

/// Fisher-Yates shuffle using the provided RNG.
fn shuffle(slice: &mut [usize], rng: &mut SmallRng) {
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(0..=i);
        slice.swap(i, j);
    }
}

// Proptest `Arbitrary` implementation for `WeightDistribution` is provided
// manually because we want biased weighting (ManyIdentical is the most
// important stress case).
impl proptest::arbitrary::Arbitrary for WeightDistribution {
    type Parameters = ();
    type Strategy = proptest::strategy::TupleUnion<(
        proptest::strategy::WA<proptest::strategy::Just<Self>>,
        proptest::strategy::WA<proptest::strategy::Just<Self>>,
        proptest::strategy::WA<proptest::strategy::Just<Self>>,
        proptest::strategy::WA<proptest::strategy::Just<Self>>,
        proptest::strategy::WA<proptest::strategy::Just<Self>>,
    )>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        prop_oneof![
            2 => Just(Self::Unique),
            3 => Just(Self::ManyIdentical),
            2 => Just(Self::Sparse),
            2 => Just(Self::Dense),
            2 => Just(Self::Disconnected),
        ]
    }
}
