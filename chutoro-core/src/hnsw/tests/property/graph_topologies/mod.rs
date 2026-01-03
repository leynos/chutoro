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

mod builders;
#[cfg(test)]
mod tests;

use builders::{
    ComponentSpec, GraphBuilder, LatticeContext, LatticeEdgeBuilder, LatticePosition,
    PreferentialAttachmentParams,
};

/// Generates an Erdos-Renyi random graph.
///
/// Each pair of nodes has probability `p` of being connected.
/// Node count ranges from 4 to 64, edge probability from 0.15 to 0.5.
/// Guarantees at least one edge by connecting nodes 0-1 when probabilistic
/// generation yields no edges.
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
/// assert!(!graph.edges.is_empty());
/// ```
pub(super) fn generate_random_graph(rng: &mut SmallRng) -> GeneratedGraph {
    let node_count = rng.gen_range(4..=64);
    // Increased minimum probability from 0.1 to 0.15 to reduce empty edge cases.
    let edge_probability = rng.gen_range(0.15..=0.5);
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

    // Guarantee at least one edge to avoid filtering in prop_filter.
    if edges.is_empty() {
        let distance = rng.gen_range(0.1_f32..10.0);
        edges.push(CandidateEdge::new(0, 1, distance, sequence));
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

    let params = PreferentialAttachmentParams::new(edges_per_new_node, exponent);
    let mut builder = GraphBuilder::new(&mut edges, &mut degrees, &mut sequence);

    // Create initial complete graph among the first `initial_nodes` nodes.
    create_initial_complete_graph(rng, &mut builder, initial_nodes, node_count);

    // Add remaining nodes with preferential attachment.
    for new_node in initial_nodes..node_count {
        attach_node_preferentially(rng, &mut builder, new_node, &params);
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

/// Creates the initial complete graph for scale-free graph generation.
///
/// Connects all pairs of nodes in the range `[0, min(initial_nodes, node_count))`
/// with random edge distances. Updates degree counts for each connection.
fn create_initial_complete_graph(
    rng: &mut SmallRng,
    builder: &mut GraphBuilder,
    initial_nodes: usize,
    node_count: usize,
) {
    for i in 0..initial_nodes.min(node_count) {
        for j in (i + 1)..initial_nodes.min(node_count) {
            let distance = rng.gen_range(0.1_f32..10.0);
            builder
                .edges
                .push(CandidateEdge::new(i, j, distance, *builder.sequence));
            builder.degrees[i] += 1;
            builder.degrees[j] += 1;
            *builder.sequence += 1;
        }
    }
}

/// Attaches a new node to the graph using preferential attachment.
///
/// Connects `new_node` to up to `edges_per_new_node` existing nodes,
/// selecting targets with probability proportional to their degree
/// raised to the given `exponent`.
fn attach_node_preferentially(
    rng: &mut SmallRng,
    builder: &mut GraphBuilder,
    new_node: usize,
    params: &PreferentialAttachmentParams,
) {
    let mut attached = Vec::new();
    for _ in 0..params.edges_per_new_node.min(new_node) {
        let target = select_by_degree(
            rng,
            &builder.degrees[..new_node],
            &attached,
            params.exponent,
        );
        if let Some(target) = target {
            let distance = rng.gen_range(0.1_f32..10.0);
            builder.edges.push(CandidateEdge::new(
                new_node,
                target,
                distance,
                *builder.sequence,
            ));
            builder.degrees[new_node] += 1;
            builder.degrees[target] += 1;
            attached.push(target);
            *builder.sequence += 1;
        }
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

    let ctx = LatticeContext::new(rows, cols, with_diagonals);
    let mut builder = LatticeEdgeBuilder::new(&mut edges, &mut sequence);

    for r in 0..rows {
        for c in 0..cols {
            let current = ctx.node_id(r, c);
            let pos = LatticePosition::new(current, r, c);
            ctx.add_edges_for_node(rng, &mut builder, &pos);
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

/// Generates a graph with disconnected components.
///
/// Creates 2-5 separate components, each with random internal structure
/// (Erdos-Renyi style within each component). Guarantees at least one edge
/// by ensuring every component has at least one internal edge.
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
/// assert!(!graph.edges.is_empty());
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
        let component_start_edge_count = edges.len();
        add_component_edges(
            rng,
            &mut edges,
            &mut sequence,
            ComponentSpec::new(node_offset, size),
        );
        // Guarantee at least one edge per component to avoid empty graphs.
        if edges.len() == component_start_edge_count && size >= 2 {
            let distance = rng.gen_range(0.1_f32..10.0);
            edges.push(CandidateEdge::new(
                node_offset,
                node_offset + 1,
                distance,
                sequence,
            ));
            sequence += 1;
        }
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
fn add_component_edges(
    rng: &mut SmallRng,
    edges: &mut Vec<CandidateEdge>,
    sequence: &mut u64,
    component: ComponentSpec,
) {
    let edge_prob = rng.gen_range(0.2..0.6);
    for i in 0..component.size {
        for j in (i + 1)..component.size {
            if !rng.gen_bool(edge_prob) {
                continue;
            }
            let distance = rng.gen_range(0.1_f32..10.0);
            edges.push(CandidateEdge::new(
                component.node_offset + i,
                component.node_offset + j,
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
    let mut last_valid = None;
    for (i, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            last_valid = Some(i);
        }
        cumulative += w;
        if cumulative >= threshold {
            return Some(i);
        }
    }
    // Fallback for floating-point precision edge case where threshold == total.
    last_valid
}
