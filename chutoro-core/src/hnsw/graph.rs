//! Internal graph representation for the CPU HNSW implementation.

use super::{
    error::HnswError,
    insert::{InsertionExecutor, InsertionPlanner},
    node::Node,
    params::HnswParams,
    search::LayerSearcher,
    types::{EntryPoint, InsertionPlan},
};

/// Context for attaching or inserting a node into the HNSW graph.
///
/// The insertion `sequence` is used for deterministic neighbour ordering and
/// trimming when distances and identifiers coincide.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeContext {
    /// Identifier of the node slot being initialised.
    pub(crate) node: usize,
    /// Highest level assigned to the node within the hierarchy.
    pub(crate) level: usize,
    /// Monotonic insertion sequence for deterministic tie-breaking.
    pub(crate) sequence: u64,
}

/// Context for connecting edges during insertion and trimming.
///
/// Encapsulates the layer level targeted by the operation alongside the
/// connection bounds applied when selecting neighbours.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeContext {
    /// Layer level for the edge operation.
    pub(crate) level: usize,
    /// Maximum number of connections permitted at this level.
    pub(crate) max_connections: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DescentContext {
    pub(crate) query: usize,
    pub(crate) target_level: usize,
    pub(crate) entry: EntryPoint,
}

impl DescentContext {
    /// Construct a descent context.
    #[must_use]
    #[inline]
    pub(crate) fn new(query: usize, entry: EntryPoint, target_level: usize) -> Self {
        Self {
            query,
            target_level,
            entry,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LayerPlanContext {
    pub(crate) query: usize,
    pub(crate) target_level: usize,
    pub(crate) current: usize,
    pub(crate) ef: usize,
}

impl LayerPlanContext {
    /// Construct a layer-planning context.
    #[must_use]
    #[inline]
    pub(crate) fn new(query: usize, current: usize, target_level: usize, ef: usize) -> Self {
        Self {
            query,
            target_level,
            current,
            ef,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ApplyContext<'a> {
    pub(crate) params: &'a HnswParams,
    pub(crate) plan: InsertionPlan,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SearchContext {
    pub(crate) query: usize,
    pub(crate) entry: usize,
    pub(crate) level: usize,
}

impl SearchContext {
    #[must_use]
    #[inline]
    pub(crate) fn with_ef(self, ef: usize) -> ExtendedSearchContext {
        ExtendedSearchContext { base: self, ef }
    }

    #[must_use]
    #[inline]
    pub(crate) fn with_distance(self, current_dist: f32) -> NeighbourSearchContext {
        NeighbourSearchContext {
            base: self,
            current_dist,
        }
    }

    #[inline]
    pub(crate) fn query(&self) -> usize {
        self.query
    }

    #[inline]
    pub(crate) fn entry(&self) -> usize {
        self.entry
    }

    #[inline]
    pub(crate) fn level(&self) -> usize {
        self.level
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ExtendedSearchContext {
    pub(crate) base: SearchContext,
    pub(crate) ef: usize,
}

impl ExtendedSearchContext {
    #[inline]
    pub(crate) fn query(&self) -> usize {
        self.base.query()
    }

    #[inline]
    pub(crate) fn entry(&self) -> usize {
        self.base.entry()
    }

    #[inline]
    pub(crate) fn level(&self) -> usize {
        self.base.level()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NeighbourSearchContext {
    base: SearchContext,
    pub(crate) current_dist: f32,
}

impl NeighbourSearchContext {
    #[inline]
    pub(crate) fn query(&self) -> usize {
        self.base.query()
    }

    #[inline]
    pub(crate) fn level(&self) -> usize {
        self.base.level()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Graph {
    params: HnswParams,
    nodes: Vec<Option<Node>>,
    entry: Option<EntryPoint>,
}

impl Graph {
    #[must_use]
    #[inline]
    pub(crate) fn with_capacity(params: HnswParams, capacity: usize) -> Self {
        debug_assert!(capacity > 0, "capacity must be greater than zero");
        Self {
            params,
            nodes: vec![None; capacity],
            entry: None,
        }
    }

    pub(crate) fn entry(&self) -> Option<EntryPoint> {
        self.entry
    }

    pub(crate) fn insert_first(&mut self, ctx: NodeContext) -> Result<(), HnswError> {
        self.attach_node(ctx)?;
        self.entry = Some(EntryPoint {
            node: ctx.node,
            level: ctx.level,
        });
        Ok(())
    }

    pub(crate) fn attach_node(&mut self, ctx: NodeContext) -> Result<(), HnswError> {
        if ctx.level > self.params.max_level() {
            return Err(HnswError::InvalidParameters {
                reason: format!(
                    "node {}: level {} exceeds max_level {}",
                    ctx.node,
                    ctx.level,
                    self.params.max_level()
                ),
            });
        }
        let slot = self
            .nodes
            .get_mut(ctx.node)
            .ok_or_else(|| HnswError::InvalidParameters {
                reason: format!("node {} is outside pre-allocated capacity", ctx.node),
            })?;
        if slot.is_some() {
            return Err(HnswError::DuplicateNode { node: ctx.node });
        }
        *slot = Some(Node::new(ctx.level, ctx.sequence));
        Ok(())
    }

    pub(crate) fn promote_entry(&mut self, node: usize, level: usize) {
        let current_level = self.entry.map(|entry| entry.level).unwrap_or(0);
        if level > current_level {
            self.entry = Some(EntryPoint { node, level });
        }
    }

    pub(crate) fn node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id).and_then(Option::as_ref)
    }

    pub(crate) fn node_mut(&mut self, id: usize) -> Option<&mut Node> {
        self.nodes.get_mut(id).and_then(Option::as_mut)
    }

    pub(crate) fn node_sequence(&self, id: usize) -> Option<u64> {
        self.node(id).map(Node::sequence)
    }

    pub(super) fn has_slot(&self, node: usize) -> bool {
        self.nodes.get(node).is_some()
    }

    #[inline]
    pub(super) fn insertion_planner(&self) -> InsertionPlanner<'_> {
        InsertionPlanner::new(self)
    }

    #[inline]
    pub(super) fn insertion_executor(&mut self) -> InsertionExecutor<'_> {
        InsertionExecutor::new(self)
    }

    #[inline]
    pub(super) fn searcher(&self) -> LayerSearcher<'_> {
        LayerSearcher::new(self)
    }

    #[cfg(test)]
    pub(super) fn params(&self) -> &HnswParams {
        &self.params
    }
}
