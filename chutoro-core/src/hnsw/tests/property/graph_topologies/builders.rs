//! Builder types for graph topology generators.
//!
//! Contains parameter objects and mutable state wrappers used to reduce
//! argument counts in graph generation helper functions.

use crate::CandidateEdge;

/// Specifies a component's position and size within a disconnected graph.
///
/// Used to reduce parameter count when adding edges within a component.
pub(super) struct ComponentSpec {
    /// Starting node index for this component.
    pub(super) node_offset: usize,
    /// Number of nodes in this component.
    pub(super) size: usize,
}

impl ComponentSpec {
    /// Creates a new component specification.
    pub(super) fn new(node_offset: usize, size: usize) -> Self {
        Self { node_offset, size }
    }
}

/// Mutable state for graph construction during scale-free generation.
///
/// Groups together the mutable references needed for adding edges,
/// reducing parameter count in helper functions.
pub(super) struct GraphBuilder<'a> {
    /// Collection of edges being built.
    pub(super) edges: &'a mut Vec<CandidateEdge>,
    /// Degree count for each node.
    pub(super) degrees: &'a mut [usize],
    /// Monotonic sequence counter for edge ordering.
    pub(super) sequence: &'a mut u64,
}

impl<'a> GraphBuilder<'a> {
    /// Creates a new graph builder from mutable references.
    pub(super) fn new(
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
pub(super) struct PreferentialAttachmentParams {
    /// Number of edges to add for each new node.
    pub(super) edges_per_new_node: usize,
    /// Exponent for the power-law degree distribution.
    pub(super) exponent: f64,
}

impl PreferentialAttachmentParams {
    /// Creates new preferential attachment parameters.
    pub(super) fn new(edges_per_new_node: usize, exponent: f64) -> Self {
        Self {
            edges_per_new_node,
            exponent,
        }
    }
}

/// Context for lattice graph generation.
///
/// Groups lattice configuration parameters to reduce argument count in
/// helper functions.
pub(super) struct LatticeContext {
    /// Number of rows in the grid.
    rows: usize,
    /// Number of columns in the grid.
    cols: usize,
    /// Whether diagonal edges are included.
    with_diagonals: bool,
}

impl LatticeContext {
    /// Creates a new lattice context.
    pub(super) fn new(rows: usize, cols: usize, with_diagonals: bool) -> Self {
        Self {
            rows,
            cols,
            with_diagonals,
        }
    }

    /// Computes the node index for a given row and column.
    pub(super) fn node_id(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    /// Adds edges for a single lattice node to the edge collection.
    pub(super) fn add_edges_for_node(
        &self,
        rng: &mut rand::rngs::SmallRng,
        builder: &mut LatticeEdgeBuilder,
        pos: &LatticePosition,
    ) {
        use rand::Rng;

        // Right neighbour.
        if pos.col + 1 < self.cols {
            let distance = rng.gen_range(0.5_f32..2.0);
            builder.add_edge(pos.current, self.node_id(pos.row, pos.col + 1), distance);
        }

        // Down neighbour.
        if pos.row + 1 < self.rows {
            let distance = rng.gen_range(0.5_f32..2.0);
            builder.add_edge(pos.current, self.node_id(pos.row + 1, pos.col), distance);
        }

        // Diagonal neighbours (if enabled).
        if !self.with_diagonals {
            return;
        }

        // Down-right diagonal.
        if pos.row + 1 < self.rows && pos.col + 1 < self.cols {
            let distance = rng.gen_range(0.7_f32..2.8);
            builder.add_edge(
                pos.current,
                self.node_id(pos.row + 1, pos.col + 1),
                distance,
            );
        }
        // Down-left diagonal.
        if pos.row + 1 < self.rows && pos.col > 0 {
            let distance = rng.gen_range(0.7_f32..2.8);
            builder.add_edge(
                pos.current,
                self.node_id(pos.row + 1, pos.col - 1),
                distance,
            );
        }
    }
}

/// Represents a position in a 2D lattice grid.
pub(super) struct LatticePosition {
    /// Current node index (computed from row and column).
    pub(super) current: usize,
    /// Row coordinate.
    pub(super) row: usize,
    /// Column coordinate.
    pub(super) col: usize,
}

impl LatticePosition {
    /// Creates a new lattice position.
    pub(super) fn new(current: usize, row: usize, col: usize) -> Self {
        Self { current, row, col }
    }
}

/// Mutable state for building lattice edges.
///
/// Groups edges and sequence counter to reduce argument count in helper methods.
pub(super) struct LatticeEdgeBuilder<'a> {
    /// Collection of edges being built.
    edges: &'a mut Vec<CandidateEdge>,
    /// Monotonic sequence counter for edge ordering.
    sequence: &'a mut u64,
}

impl<'a> LatticeEdgeBuilder<'a> {
    /// Creates a new lattice edge builder.
    pub(super) fn new(edges: &'a mut Vec<CandidateEdge>, sequence: &'a mut u64) -> Self {
        Self { edges, sequence }
    }

    /// Adds an edge to the collection with the current sequence number.
    pub(super) fn add_edge(&mut self, source: usize, target: usize, distance: f32) {
        self.edges
            .push(CandidateEdge::new(source, target, distance, *self.sequence));
        *self.sequence += 1;
    }
}
