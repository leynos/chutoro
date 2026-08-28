//! Shared types used across the insertion executor submodules.
//!
//! These definitions model the staged state, trimming work, and per-level
//! contexts required while applying an insertion. They are kept in a dedicated
//! module to avoid visibility tangles between sibling components such as
//! staging, reconciliation, and connectivity healing.

use std::collections::{HashMap, HashSet};

use crate::hnsw::graph::{EdgeContext, NodeContext};

/// Captures the neighbour candidates for a node that may require trimming.
///
/// Each candidate has a corresponding insertion sequence used to implement the
/// deterministic tie-break when trimming applies.
///
/// # Examples
/// ```rust,ignore
/// use crate::hnsw::{
///     graph::EdgeContext,
///     insert::TrimJob,
/// };
///
/// let ctx = EdgeContext { level: 0, max_connections: 2 };
/// let job = TrimJob {
///     node: 1,
///     ctx,
///     candidates: vec![2, 3],
///     sequences: vec![4, 5],
/// };
/// assert_eq!(job.candidates.len(), job.sequences.len());
/// ```
#[derive(Clone, Debug)]
pub(crate) struct TrimJob {
    /// Node whose neighbours are being trimmed.
    pub(crate) node: usize,
    /// Layer and degree bound applied to the trim.
    pub(crate) ctx: EdgeContext,
    /// Candidate neighbour identifiers before trimming.
    pub(crate) candidates: Vec<usize>,
    /// Insertion sequences aligned with the candidates.
    pub(crate) sequences: Vec<u64>,
}

/// Fully prepared graph mutation before its staged updates are applied.
#[derive(Clone, Debug)]
pub(crate) struct PreparedInsertion {
    /// Context describing the newly inserted node.
    pub(crate) node: NodeContext,
    /// Whether this node becomes the graph entry point.
    pub(crate) promote_entry: bool,
    /// Neighbours selected for the new node at each layer.
    pub(crate) new_node_neighbours: Vec<Vec<usize>>,
    /// Existing-node neighbour updates staged by the insertion.
    pub(crate) updates: Vec<StagedUpdate>,
    /// Degree bound for the level-zero insertion updates.
    pub(crate) max_connections: usize,
}

/// Captures the staged neighbour set for a node at a given level.
#[derive(Clone, Debug)]
pub(crate) struct StagedUpdate {
    /// Existing node whose neighbours will be replaced.
    pub(crate) node: usize,
    /// Layer and degree bound for this update.
    pub(crate) ctx: EdgeContext,
    /// Candidate neighbours before deterministic trimming.
    pub(crate) candidates: Vec<usize>,
}

/// Stores the final trimmed neighbour list for a node and level.
#[derive(Clone, Debug)]
pub(crate) struct TrimResult {
    /// Node whose neighbours were trimmed.
    pub(crate) node: usize,
    /// Layer and degree bound used during trimming.
    pub(crate) ctx: EdgeContext,
    /// Final neighbour identifiers after trimming.
    pub(crate) neighbours: Vec<usize>,
}

/// A staged update paired with its final trimmed neighbours.
pub(crate) type FinalisedUpdate = (StagedUpdate, Vec<usize>);

/// Outcome of staging the insertion layers prior to trimming.
pub(super) struct LayerProcessingOutcome {
    /// Per-layer neighbours selected for the inserted node.
    pub(super) new_node_neighbours: Vec<Vec<usize>>,
    /// Staged neighbour lists keyed by origin and layer.
    pub(super) staged: HashMap<(usize, usize), Vec<usize>>,
    /// Node-layer pairs initialised during staging.
    pub(super) initialised: HashSet<(usize, usize)>,
    /// Node-layer pairs that exceed their degree limit.
    pub(super) needs_trim: HashSet<(usize, usize)>,
}

/// Accumulates the staged neighbour lists and trimming metadata.
pub(super) struct TrimWork {
    /// Staged neighbour lists keyed by origin and layer.
    pub(super) staged: HashMap<(usize, usize), Vec<usize>>,
    /// Node-layer pairs requiring a trim operation.
    pub(super) needs_trim: HashSet<(usize, usize)>,
    /// Degree bound applied to the accumulated updates.
    pub(super) max_connections: usize,
}

/// Identifier and level assigned to a newly inserted node.
#[derive(Clone, Copy)]
pub(crate) struct NewNodeContext {
    /// Identifier of the new graph node.
    pub(crate) id: usize,
    /// Highest HNSW level assigned to the new node.
    pub(crate) level: usize,
}

/// Origin and bounds for an existing-node neighbour update.
#[derive(Clone, Copy)]
pub(super) struct UpdateContext {
    /// Existing node whose neighbours are updated.
    pub(super) origin: usize,
    /// HNSW layer receiving the update.
    pub(super) level: usize,
    /// Maximum degree allowed at this layer.
    pub(super) max_connections: usize,
}

/// Layer and bounds for linking a new node to an existing neighbour.
#[derive(Clone, Copy)]
pub(super) struct LinkContext {
    /// HNSW layer receiving the reciprocal link.
    pub(super) level: usize,
    /// Maximum degree allowed at this layer.
    pub(super) max_connections: usize,
    /// Newly inserted node being linked.
    pub(super) new_node: usize,
}

/// Context for healing connectivity gaps during insertion.
pub(super) struct HealingContext<'a> {
    /// New-node neighbours retained after filtering invalid links.
    pub(super) filtered_new_node_neighbours: &'a [Vec<usize>],
    /// Identifier of the newly inserted node.
    pub(super) new_node_id: usize,
    /// Degree bound used by connectivity healing.
    pub(super) max_connections: usize,
}

/// A deferred scrub request collected during reconciliation.
///
/// When `ensure_reverse_edge` evicts a node to make room for a new link, we
/// cannot immediately scrub the forward edge because a later update in the same
/// batch might re-add it. Instead, scrub requests are collected and filtered
/// before application.
#[derive(Clone, Debug)]
pub(super) struct DeferredScrub {
    /// The node whose forward edge should be scrubbed.
    pub(super) origin: usize,
    /// The target of the forward edge to remove.
    pub(super) target: usize,
    /// The level at which the edge exists.
    pub(super) level: usize,
}
