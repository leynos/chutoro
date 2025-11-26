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
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) candidates: Vec<usize>,
    pub(crate) sequences: Vec<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedInsertion {
    pub(crate) node: NodeContext,
    pub(crate) promote_entry: bool,
    pub(crate) new_node_neighbours: Vec<Vec<usize>>,
    pub(crate) updates: Vec<StagedUpdate>,
    pub(crate) max_connections: usize,
}

/// Captures the staged neighbour set for a node at a given level.
#[derive(Clone, Debug)]
pub(crate) struct StagedUpdate {
    pub(super) node: usize,
    pub(super) ctx: EdgeContext,
    pub(super) candidates: Vec<usize>,
}

/// Stores the final trimmed neighbour list for a node and level.
#[derive(Clone, Debug)]
pub(crate) struct TrimResult {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) neighbours: Vec<usize>,
}

pub(super) type FinalisedUpdate = (StagedUpdate, Vec<usize>);

/// Outcome of staging the insertion layers prior to trimming.
pub(super) type LayerProcessingOutcome = (
    Vec<Vec<usize>>,
    HashMap<(usize, usize), Vec<usize>>,
    HashSet<(usize, usize)>,
    HashSet<(usize, usize)>,
);

/// Accumulates the staged neighbour lists and trimming metadata.
pub(super) struct TrimWork {
    pub(super) staged: HashMap<(usize, usize), Vec<usize>>,
    pub(super) needs_trim: HashSet<(usize, usize)>,
    pub(super) max_connections: usize,
}

#[derive(Clone, Copy)]
pub(super) struct NewNodeContext {
    pub(super) id: usize,
    pub(super) level: usize,
}

#[derive(Clone, Copy)]
pub(super) struct UpdateContext {
    pub(super) origin: usize,
    pub(super) level: usize,
    pub(super) max_connections: usize,
}

#[derive(Clone, Copy)]
pub(super) struct LinkContext {
    pub(super) level: usize,
    pub(super) max_connections: usize,
    pub(super) new_node: usize,
}

/// Context for healing connectivity gaps during insertion.
pub(super) struct HealingContext<'a> {
    pub filtered_new_node_neighbours: &'a [Vec<usize>],
    pub new_node_id: usize,
    pub max_connections: usize,
}
