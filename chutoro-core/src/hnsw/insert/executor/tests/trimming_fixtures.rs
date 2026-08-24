//! Test fixtures and helpers for trimming eviction scenarios.
//!
//! These helpers build graphs with specific neighbour configurations to
//! exercise the insertion executor's trimming and reciprocity restoration
//! logic.

use super::*;
use crate::hnsw::{
    error::HnswError,
    graph::{ApplyContext, Graph, NodeContext},
    params::{HnswParams, connection_limit_for_level},
    types::{InsertionPlan, LayerPlan, Neighbour},
};

/// Failure modes of the trimming fixtures.
///
/// Keeps the underlying [`HnswError`] intact while naming the fixture
/// invariants that the helpers rely on, so a broken fixture is distinguishable
/// from a genuine graph error.
#[derive(Debug, thiserror::Error)]
pub(super) enum TrimmingFixtureError {
    /// The trimming scenario requires at least one neighbour.
    #[error("trimmed neighbours must be non-empty")]
    EmptyTrimmedNeighbours,
    /// A graph or executor operation failed.
    #[error(transparent)]
    Hnsw(#[from] HnswError),
    /// The executor produced no trim job for the entry node.
    #[error("trim job expected")]
    MissingTrimJob,
    /// The executor produced more trim jobs than the fixture expects.
    #[error("only the entry node should require trim, but {count} jobs were produced")]
    ExcessTrimJobs {
        /// Number of trim jobs the executor returned.
        count: usize,
    },
    /// The executor trimmed a node other than the entry node.
    #[error("trim must target the entry node, but node {node} was targeted")]
    UnexpectedTrimTarget {
        /// Identifier of the node the executor actually trimmed.
        node: usize,
    },
    /// A node the fixture requires was absent from the graph.
    #[error("node {node} should be present: {reason}")]
    MissingNode {
        /// Identifier of the absent node.
        node: usize,
        /// Why the fixture expected the node to exist.
        reason: &'static str,
    },
    /// A post-trim reciprocity invariant did not hold.
    #[error("{invariant}")]
    ReciprocityViolated {
        /// Description of the invariant that failed.
        invariant: &'static str,
    },
}

/// Returns [`TrimmingFixtureError::ReciprocityViolated`] unless `is_satisfied`.
///
/// # Errors
/// Returns the named invariant as an error so the calling test reports the
/// failure rather than the helper panicking.
fn ensure_invariant(
    is_satisfied: bool,
    invariant: &'static str,
) -> Result<(), TrimmingFixtureError> {
    if is_satisfied {
        return Ok(());
    }
    Err(TrimmingFixtureError::ReciprocityViolated { invariant })
}

pub(super) fn apply_insertion_with_trim(
    graph: &mut Graph,
    params: &HnswParams,
    new_node_id: usize,
    trimmed_neighbours: Vec<usize>,
) -> Result<(), TrimmingFixtureError> {
    let reserve_id = new_node_id.saturating_add(1);
    let mut executor = InsertionExecutor::new(graph);
    let plan = InsertionPlan {
        layers: vec![LayerPlan {
            level: 0,
            neighbours: vec![Neighbour {
                id: 0,
                distance: 0.0,
            }],
        }],
    };

    let (prepared, trim_jobs) = executor.apply(
        NodeContext {
            node: new_node_id,
            level: 0,
            sequence: (reserve_id as u64) + 1,
        },
        ApplyContext { params, plan },
    )?;

    let mut jobs = trim_jobs.into_iter();
    let job = jobs.next().ok_or(TrimmingFixtureError::MissingTrimJob)?;
    let excess = jobs.count();
    if excess > 0 {
        return Err(TrimmingFixtureError::ExcessTrimJobs {
            count: excess.saturating_add(1),
        });
    }
    if job.node != 0 {
        return Err(TrimmingFixtureError::UnexpectedTrimTarget { node: job.node });
    }
    let trim_result = TrimResult {
        node: job.node,
        ctx: job.ctx,
        neighbours: trimmed_neighbours,
    };

    Ok(executor.commit(prepared, vec![trim_result])?)
}

pub(super) fn verify_post_trim_reciprocity(
    graph: &Graph,
    params: &HnswParams,
    new_node_id: usize,
    evicted: usize,
) -> Result<(), TrimmingFixtureError> {
    let connection_limit = connection_limit_for_level(0, params.max_connections());
    let entry = graph.node(0).ok_or(TrimmingFixtureError::MissingNode {
        node: 0,
        reason: "entry node available",
    })?;
    let entry_neighbours = entry.neighbours(0);
    ensure_invariant(
        entry_neighbours.contains(&new_node_id),
        "reciprocity pass should reintroduce the new node even after trim eviction",
    )?;
    ensure_invariant(
        !entry_neighbours.contains(&evicted),
        "evicted neighbour should be removed to honour capacity constraints",
    )?;
    ensure_invariant(
        entry_neighbours.len() <= connection_limit,
        "entry degree should respect the base-layer limit after reconciliation",
    )?;

    let new_node = graph
        .node(new_node_id)
        .ok_or(TrimmingFixtureError::MissingNode {
            node: new_node_id,
            reason: "new node must be attached after commit",
        })?;
    ensure_invariant(
        new_node.neighbours(0).contains(&0),
        "new node should have a reciprocal edge to the entry node",
    )?;

    let evicted_node = graph
        .node(evicted)
        .ok_or(TrimmingFixtureError::MissingNode {
            node: evicted,
            reason: "evicted neighbour must remain attached after commit",
        })?;
    ensure_invariant(
        !evicted_node.neighbours(0).contains(&0),
        "forward edge from evicted neighbour should be removed",
    )?;
    Ok(())
}

pub(super) fn build_trimming_test_graph(
    params: &HnswParams,
    trimmed_neighbours: &[usize],
    reserve_id: usize,
) -> Result<Graph, TrimmingFixtureError> {
    let capacity = reserve_id.saturating_add(1);
    let mut graph = Graph::with_capacity(params.clone(), capacity);
    graph.insert_first(NodeContext {
        node: 0,
        level: 0,
        sequence: 0,
    })?;

    attach_neighbours(&mut graph, trimmed_neighbours, 1)?;
    graph.attach_node(NodeContext {
        node: reserve_id,
        level: 0,
        sequence: (trimmed_neighbours.len() + 1) as u64,
    })?;

    set_entry_neighbours(&mut graph, trimmed_neighbours)?;

    Ok(graph)
}

fn attach_neighbours(
    graph: &mut Graph,
    neighbours: &[usize],
    start_sequence: u64,
) -> Result<(), HnswError> {
    for (offset, &id) in neighbours.iter().enumerate() {
        graph.attach_node(NodeContext {
            node: id,
            level: 0,
            sequence: start_sequence + offset as u64,
        })?;
    }
    Ok(())
}

pub(super) fn setup_reciprocal_edges_with_reserve(
    graph: &mut Graph,
    trimmed_neighbours: &[usize],
    evicted: usize,
    reserve_id: usize,
) -> Result<(), TrimmingFixtureError> {
    set_entry_neighbours(graph, trimmed_neighbours)?;

    for &neighbour in trimmed_neighbours {
        link_if_absent(graph, neighbour, 0)?;
    }

    link_if_absent(graph, evicted, reserve_id)?;
    link_if_absent(graph, reserve_id, evicted)?;
    Ok(())
}

fn set_entry_neighbours(
    graph: &mut Graph,
    neighbours: &[usize],
) -> Result<(), TrimmingFixtureError> {
    let entry_neighbours = graph
        .node_mut(0)
        .ok_or(TrimmingFixtureError::MissingNode {
            node: 0,
            reason: "entry present",
        })?
        .neighbours_mut(0);
    entry_neighbours.clear();
    entry_neighbours.extend(neighbours.iter().copied());
    Ok(())
}

/// Adds a base-layer edge from `origin` to `target` unless it already exists.
///
/// # Errors
/// Returns [`TrimmingFixtureError::MissingNode`] if `origin` is not attached to
/// the graph.
fn link_if_absent(
    graph: &mut Graph,
    origin: usize,
    target: usize,
) -> Result<(), TrimmingFixtureError> {
    let node = graph
        .node_mut(origin)
        .ok_or(TrimmingFixtureError::MissingNode {
            node: origin,
            reason: "link origin must be attached before linking",
        })?;
    let list = node.neighbours_mut(0);
    if !list.contains(&target) {
        list.push(target);
    }
    Ok(())
}
