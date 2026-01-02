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

#[cfg(test)]
mod tests;

/// Specifies a component's position and size within a disconnected graph.
///
/// Used to reduce parameter count when adding edges within a component.
struct ComponentSpec {
    /// Starting node index for this component.
    node_offset: usize,
    /// Number of nodes in this component.
    size: usize,
}

impl ComponentSpec {
    /// Creates a new component specification.
    fn new(node_offset: usize, size: usize) -> Self {
        Self { node_offset, size }
    }
}

/// Mutable state for graph construction during scale-free generation.
///
/// Groups together the mutable references needed for adding edges,
/// reducing parameter count in helper functions.
struct GraphBuilder<'a> {
    /// Collection of edges being built.
    edges: &'a mut Vec<CandidateEdge>,
    /// Degree count for each node.
    degrees: &'a mut [usize],
    /// Monotonic sequence counter for edge ordering.
    sequence: &'a mut u64,
}

impl<'a> GraphBuilder<'a> {
    /// Creates a new graph builder from mutable references.
    fn new(
        edges: &'a mut Vec<CandidateEdge>,
        degrees: &'a mut [usize],
        sequence: &'a mut u64,
    ) -> Self {
        Self {
            edges,
            degrees,
            sequence,
        }
    }
}

/// Parameters for preferential attachment in scale-free graph generation.
///
/// Groups algorithm parameters to reduce argument count in helper functions.
struct PreferentialAttachmentParams {
    /// Number of edges to add for each new node.
    edges_per_new_node: usize,
    /// Exponent for the power-law degree distribution.
    exponent: f64,
}

impl PreferentialAttachmentParams {
    /// Creates new preferential attachment parameters.
    fn new(edges_per_new_node: usize, exponent: f64) -> Self {
        Self {
            edges_per_new_node,
            exponent,
        }
    }
}

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

/// Context for lattice graph generation.
///
/// Groups lattice configuration parameters to reduce argument count in
/// helper functions.
struct LatticeContext {
    /// Number of rows in the grid.
    rows: usize,
    /// Number of columns in the grid.
    cols: usize,
    /// Whether diagonal edges are included.
    with_diagonals: bool,
}

impl LatticeContext {
    /// Creates a new lattice context.
    fn new(rows: usize, cols: usize, with_diagonals: bool) -> Self {
        Self {
            rows,
            cols,
            with_diagonals,
        }
    }

    /// Computes the node index for a given row and column.
    fn node_id(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    /// Adds edges for a single lattice node to the edge collection.
    #[expect(
        clippy::too_many_arguments,
        reason = "uses parameter objects for context; remaining args are mutable state and rng"
    )]
    fn add_edges_for_node(
        &self,
        rng: &mut SmallRng,
        edges: &mut Vec<CandidateEdge>,
        sequence: &mut u64,
        pos: &LatticePosition,
    ) {
        // Right neighbour.
        if pos.col + 1 < self.cols {
            let distance = rng.gen_range(0.5_f32..2.0);
            edges.push(CandidateEdge::new(
                pos.current,
                self.node_id(pos.row, pos.col + 1),
                distance,
                *sequence,
            ));
            *sequence += 1;
        }

        // Down neighbour.
        if pos.row + 1 < self.rows {
            let distance = rng.gen_range(0.5_f32..2.0);
            edges.push(CandidateEdge::new(
                pos.current,
                self.node_id(pos.row + 1, pos.col),
                distance,
                *sequence,
            ));
            *sequence += 1;
        }

        // Diagonal neighbours (if enabled).
        if !self.with_diagonals {
            return;
        }

        // Down-right diagonal.
        if pos.row + 1 < self.rows && pos.col + 1 < self.cols {
            let distance = rng.gen_range(0.7_f32..2.8);
            edges.push(CandidateEdge::new(
                pos.current,
                self.node_id(pos.row + 1, pos.col + 1),
                distance,
                *sequence,
            ));
            *sequence += 1;
        }
        // Down-left diagonal.
        if pos.row + 1 < self.rows && pos.col > 0 {
            let distance = rng.gen_range(0.7_f32..2.8);
            edges.push(CandidateEdge::new(
                pos.current,
                self.node_id(pos.row + 1, pos.col - 1),
                distance,
                *sequence,
            ));
            *sequence += 1;
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

    for r in 0..rows {
        for c in 0..cols {
            let current = ctx.node_id(r, c);
            let pos = LatticePosition::new(current, r, c);
            ctx.add_edges_for_node(rng, &mut edges, &mut sequence, &pos);
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

/// Represents a position in a 2D lattice grid.
struct LatticePosition {
    /// Current node index (computed from row and column).
    current: usize,
    /// Row coordinate.
    row: usize,
    /// Column coordinate.
    col: usize,
}

impl LatticePosition {
    /// Creates a new lattice position.
    fn new(current: usize, row: usize, col: usize) -> Self {
        Self { current, row, col }
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
    for (i, w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative >= threshold {
            return Some(i);
        }
    }
    None
}
