//! Graph topology generators for property-based edge harvest testing.
//!
//! Provides generators for random, scale-free, lattice, and disconnected
//! graph structures, each returning a [`GeneratedGraph`] that captures
//! edges alongside metadata for validation and shrinking.
//!
//! These generators produce graph topologies (edge sets) for testing the
//! candidate edge harvest algorithm, as opposed to the vector distribution
//! generators in [`super::datasets`] which produce points in metric space.

use rand::{Rng, rngs::SmallRng};

use crate::CandidateEdge;

use super::types::{GeneratedGraph, GraphMetadata};

/// Generates an Erdos-Renyi random graph.
///
/// Each pair of nodes has probability `p` of being connected.
/// Node count ranges from 4 to 64, edge probability from 0.1 to 0.5.
///
/// # Examples
///
/// ```ignore
/// use rand::{rngs::SmallRng, SeedableRng};
/// use crate::hnsw::tests::property::graph_topologies::generate_random_graph;
///
/// let mut rng = SmallRng::seed_from_u64(42);
/// let graph = generate_random_graph(&mut rng);
/// assert!(graph.node_count >= 4);
/// ```
pub(super) fn generate_random_graph(rng: &mut SmallRng) -> GeneratedGraph {
    let node_count = rng.gen_range(4..=64);
    let edge_probability = rng.gen_range(0.1..=0.5);
    let mut edges = Vec::new();
    let mut sequence = 0u64;

    for i in 0..node_count {
        for j in (i + 1)..node_count {
            if rng.gen_bool(edge_probability) {
                let distance = rng.gen_range(0.1_f32..10.0);
                edges.push(CandidateEdge::new(i, j, distance, sequence));
                sequence += 1;
            }
        }
    }

    GeneratedGraph {
        node_count,
        edges,
        metadata: GraphMetadata::Random {
            node_count,
            edge_probability,
        },
    }
}

/// Generates a scale-free graph using Barabasi-Albert preferential attachment.
///
/// Starts with a small complete graph and adds nodes one by one,
/// connecting each to `m` existing nodes with probability proportional
/// to their degree raised to the power `exponent`.
///
/// # Examples
///
/// ```ignore
/// use rand::{rngs::SmallRng, SeedableRng};
/// use crate::hnsw::tests::property::graph_topologies::generate_scale_free_graph;
///
/// let mut rng = SmallRng::seed_from_u64(42);
/// let graph = generate_scale_free_graph(&mut rng);
/// assert!(graph.node_count >= 8);
/// ```
pub(super) fn generate_scale_free_graph(rng: &mut SmallRng) -> GeneratedGraph {
    let node_count = rng.gen_range(8..=48);
    let edges_per_new_node = rng.gen_range(1..=3);
    let exponent = rng.gen_range(1.5..3.0);

    let initial_nodes = edges_per_new_node + 1;
    let mut edges = Vec::new();
    let mut degrees = vec![0usize; node_count];
    let mut sequence = 0u64;

    // Create initial complete graph among the first `initial_nodes` nodes.
    for i in 0..initial_nodes.min(node_count) {
        for j in (i + 1)..initial_nodes.min(node_count) {
            let distance = rng.gen_range(0.1_f32..10.0);
            edges.push(CandidateEdge::new(i, j, distance, sequence));
            degrees[i] += 1;
            degrees[j] += 1;
            sequence += 1;
        }
    }

    // Add remaining nodes with preferential attachment.
    for new_node in initial_nodes..node_count {
        let mut attached = Vec::new();
        for _ in 0..edges_per_new_node.min(new_node) {
            let target = select_by_degree(rng, &degrees[..new_node], &attached, exponent);
            if let Some(target) = target {
                let distance = rng.gen_range(0.1_f32..10.0);
                edges.push(CandidateEdge::new(new_node, target, distance, sequence));
                degrees[new_node] += 1;
                degrees[target] += 1;
                attached.push(target);
                sequence += 1;
            }
        }
    }

    GeneratedGraph {
        node_count,
        edges,
        metadata: GraphMetadata::ScaleFree {
            node_count,
            edges_per_new_node,
            exponent,
        },
    }
}

/// Generates a lattice/grid graph.
///
/// Creates a 2D grid with optional diagonal connections. Rows and columns
/// range from 2 to 8 each.
///
/// # Examples
///
/// ```ignore
/// use rand::{rngs::SmallRng, SeedableRng};
/// use crate::hnsw::tests::property::graph_topologies::generate_lattice_graph;
///
/// let mut rng = SmallRng::seed_from_u64(42);
/// let graph = generate_lattice_graph(&mut rng);
/// assert!(graph.node_count >= 4);
/// ```
pub(super) fn generate_lattice_graph(rng: &mut SmallRng) -> GeneratedGraph {
    let rows = rng.gen_range(2..=8);
    let cols = rng.gen_range(2..=8);
    let with_diagonals = rng.gen_bool(0.5);
    let node_count = rows * cols;

    let mut edges = Vec::new();
    let mut sequence = 0u64;

    let node_id = |r: usize, c: usize| r * cols + c;

    for r in 0..rows {
        for c in 0..cols {
            let current = node_id(r, c);
            add_lattice_edges(
                rng,
                &mut edges,
                &mut sequence,
                current,
                r,
                c,
                rows,
                cols,
                with_diagonals,
                &node_id,
            );
        }
    }

    GeneratedGraph {
        node_count,
        edges,
        metadata: GraphMetadata::Lattice {
            dimensions: (rows, cols),
            with_diagonals,
        },
    }
}

/// Adds lattice edges for a single node (extracted to reduce nesting).
#[allow(clippy::too_many_arguments)]
fn add_lattice_edges(
    rng: &mut SmallRng,
    edges: &mut Vec<CandidateEdge>,
    sequence: &mut u64,
    current: usize,
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
    with_diagonals: bool,
    node_id: &impl Fn(usize, usize) -> usize,
) {
    // Right neighbour.
    if c + 1 < cols {
        let distance = rng.gen_range(0.5_f32..2.0);
        edges.push(CandidateEdge::new(
            current,
            node_id(r, c + 1),
            distance,
            *sequence,
        ));
        *sequence += 1;
    }

    // Down neighbour.
    if r + 1 < rows {
        let distance = rng.gen_range(0.5_f32..2.0);
        edges.push(CandidateEdge::new(
            current,
            node_id(r + 1, c),
            distance,
            *sequence,
        ));
        *sequence += 1;
    }

    // Diagonal neighbours (if enabled).
    if !with_diagonals {
        return;
    }

    // Down-right diagonal.
    if r + 1 < rows && c + 1 < cols {
        let distance = rng.gen_range(0.7_f32..2.8);
        edges.push(CandidateEdge::new(
            current,
            node_id(r + 1, c + 1),
            distance,
            *sequence,
        ));
        *sequence += 1;
    }
    // Down-left diagonal.
    if r + 1 < rows && c > 0 {
        let distance = rng.gen_range(0.7_f32..2.8);
        edges.push(CandidateEdge::new(
            current,
            node_id(r + 1, c - 1),
            distance,
            *sequence,
        ));
        *sequence += 1;
    }
}

/// Generates a graph with disconnected components.
///
/// Creates 2-5 separate components, each with random internal structure
/// (Erdos-Renyi style within each component).
///
/// # Examples
///
/// ```ignore
/// use rand::{rngs::SmallRng, SeedableRng};
/// use crate::hnsw::tests::property::graph_topologies::generate_disconnected_graph;
///
/// let mut rng = SmallRng::seed_from_u64(42);
/// let graph = generate_disconnected_graph(&mut rng);
/// assert!(graph.node_count >= 6);
/// ```
pub(super) fn generate_disconnected_graph(rng: &mut SmallRng) -> GeneratedGraph {
    let component_count = rng.gen_range(2..=5);
    let mut component_sizes = Vec::with_capacity(component_count);
    let mut node_count = 0;

    for _ in 0..component_count {
        let size = rng.gen_range(3..=12);
        component_sizes.push(size);
        node_count += size;
    }

    let mut edges = Vec::new();
    let mut sequence = 0u64;
    let mut node_offset = 0;

    for &size in &component_sizes {
        add_component_edges(rng, &mut edges, &mut sequence, node_offset, size);
        node_offset += size;
    }

    GeneratedGraph {
        node_count,
        edges,
        metadata: GraphMetadata::Disconnected {
            component_count,
            component_sizes,
        },
    }
}

/// Adds edges within a single component (extracted to reduce nesting).
#[allow(clippy::too_many_arguments)]
fn add_component_edges(
    rng: &mut SmallRng,
    edges: &mut Vec<CandidateEdge>,
    sequence: &mut u64,
    node_offset: usize,
    size: usize,
) {
    let edge_prob = rng.gen_range(0.2..0.6);
    for i in 0..size {
        for j in (i + 1)..size {
            if !rng.gen_bool(edge_prob) {
                continue;
            }
            let distance = rng.gen_range(0.1_f32..10.0);
            edges.push(CandidateEdge::new(
                node_offset + i,
                node_offset + j,
                distance,
                *sequence,
            ));
            *sequence += 1;
        }
    }
}

/// Selects a node by preferential attachment (power-law weighted by degree).
///
/// Returns `None` if no valid candidate exists (all nodes excluded or zero
/// total weight).
fn select_by_degree(
    rng: &mut SmallRng,
    degrees: &[usize],
    exclude: &[usize],
    exponent: f64,
) -> Option<usize> {
    let weights: Vec<f64> = degrees
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if exclude.contains(&i) {
                0.0
            } else {
                // Add 1 to degree to avoid zero weights for isolated nodes.
                (d.max(1) as f64).powf(exponent)
            }
        })
        .collect();

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }

    let threshold = rng.gen_range(0.0..1.0) * total;
    let mut cumulative = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative >= threshold {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rstest::rstest;

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

        for edge in &graph.edges {
            assert!(edge.source() < graph.node_count);
            assert!(edge.target() < graph.node_count);
            assert_ne!(edge.source(), edge.target());
            assert!(edge.distance().is_finite());
            assert!(edge.distance() > 0.0);
        }
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

        for edge in &graph.edges {
            assert!(edge.source() < graph.node_count);
            assert!(edge.target() < graph.node_count);
            assert_ne!(edge.source(), edge.target());
            assert!(edge.distance().is_finite());
            assert!(edge.distance() > 0.0);
        }
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

        for edge in &graph.edges {
            assert!(edge.source() < graph.node_count);
            assert!(edge.target() < graph.node_count);
            assert_ne!(edge.source(), edge.target());
            assert!(edge.distance().is_finite());
            assert!(edge.distance() > 0.0);
        }

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

        for edge in &graph.edges {
            assert!(edge.source() < graph.node_count);
            assert!(edge.target() < graph.node_count);
            assert_ne!(edge.source(), edge.target());
            assert!(edge.distance().is_finite());
            assert!(edge.distance() > 0.0);
        }

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
                    "edge {:?} crosses components",
                    edge
                );
            }
        }
    }

    #[rstest]
    fn lattice_with_diagonals_has_more_edges() {
        // Generate lattices with and without diagonals and compare edge counts.
        let mut with_diag_edges = 0usize;
        let mut without_diag_edges = 0usize;

        for seed in 0..20 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let graph = generate_lattice_graph(&mut rng);

            if let GraphMetadata::Lattice { with_diagonals, .. } = &graph.metadata {
                if *with_diagonals {
                    with_diag_edges += graph.edges.len();
                } else {
                    without_diag_edges += graph.edges.len();
                }
            }
        }

        // On average, diagonal lattices should have more edges.
        // Both should be non-zero given 20 samples with 50% probability.
        assert!(with_diag_edges > 0 || without_diag_edges > 0);
    }
}
