//! Internal graph representation for the CPU HNSW implementation.

use crate::hnsw::{
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
    /// Identifier of the node slot being initialized.
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

/// Context for descending from an entry point to a target HNSW level.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DescentContext {
    /// Node used as the distance query during descent.
    pub(crate) query: usize,
    /// Lowest layer included in the descent.
    pub(crate) target_level: usize,
    /// Starting node and layer for descent.
    pub(crate) entry: EntryPoint,
}

impl DescentContext {
    /// Construct a descent context.
    #[must_use]
    #[inline]
    pub(crate) const fn new(query: usize, entry: EntryPoint, target_level: usize) -> Self {
        Self {
            query,
            target_level,
            entry,
        }
    }
}

/// Inputs needed to plan neighbours for a single HNSW layer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerPlanContext {
    /// Node used as the distance query.
    pub(crate) query: usize,
    /// HNSW layer being planned.
    pub(crate) target_level: usize,
    /// Current entry node for the layer search.
    pub(crate) current: usize,
    /// Candidate-set width used during planning.
    pub(crate) ef: usize,
}

impl LayerPlanContext {
    /// Construct a layer-planning context.
    #[must_use]
    #[inline]
    pub(crate) const fn new(query: usize, current: usize, target_level: usize, ef: usize) -> Self {
        Self {
            query,
            target_level,
            current,
            ef,
        }
    }
}

/// Inputs needed to apply an insertion plan to a graph.
#[derive(Clone, Debug)]
pub(crate) struct ApplyContext<'a> {
    /// Connection bounds that govern plan application.
    pub(crate) params: &'a HnswParams,
    /// Staged insertion changes to commit.
    pub(crate) plan: InsertionPlan,
}

/// Query, entry, and layer inputs shared by HNSW searches.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SearchContext {
    /// Node used as the distance query.
    pub(crate) query: usize,
    /// Node from which the layer search starts.
    pub(crate) entry: usize,
    /// HNSW layer searched by this operation.
    pub(crate) level: usize,
}

impl SearchContext {
    /// Extend this context with a candidate-set width.
    #[must_use]
    #[inline]
    pub(crate) const fn with_ef(self, ef: usize) -> ExtendedSearchContext {
        ExtendedSearchContext { base: self, ef }
    }

    /// Extend this context with the current entry distance.
    #[must_use]
    #[inline]
    pub(crate) const fn with_distance(self, current_dist: f32) -> NeighbourSearchContext {
        NeighbourSearchContext {
            base: self,
            current_dist,
        }
    }

    /// Return the query node identifier.
    #[inline]
    pub(crate) const fn query(&self) -> usize {
        self.query
    }

    /// Return the entry node identifier.
    #[inline]
    pub(crate) const fn entry(&self) -> usize {
        self.entry
    }

    /// Return the HNSW layer identifier.
    #[inline]
    pub(crate) const fn level(&self) -> usize {
        self.level
    }
}

/// Search context extended with its candidate-set width.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExtendedSearchContext {
    /// Common query, entry, and layer inputs.
    pub(crate) base: SearchContext,
    /// Candidate-set width applied to the search.
    pub(crate) ef: usize,
}

impl ExtendedSearchContext {
    /// Return the query node identifier.
    #[inline]
    pub(crate) const fn query(&self) -> usize {
        self.base.query()
    }

    /// Return the entry node identifier.
    #[inline]
    pub(crate) const fn entry(&self) -> usize {
        self.base.entry()
    }

    /// Return the HNSW layer identifier.
    #[inline]
    pub(crate) const fn level(&self) -> usize {
        self.base.level()
    }
}

/// Search context extended with the current entry distance.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NeighbourSearchContext {
    /// Common query, entry, and layer inputs.
    base: SearchContext,
    /// Distance from the query to the current entry node.
    pub(crate) current_dist: f32,
}

impl NeighbourSearchContext {
    /// Return the query node identifier.
    #[inline]
    pub(crate) const fn query(&self) -> usize {
        self.base.query()
    }

    /// Return the HNSW layer identifier.
    #[inline]
    pub(crate) const fn level(&self) -> usize {
        self.base.level()
    }
}

/// Preallocated HNSW graph state and its current entry point.
#[derive(Clone, Debug)]
pub(crate) struct Graph {
    /// Immutable connection and layer constraints for this graph.
    pub(super) params: HnswParams,
    /// Preallocated node slots, populated as nodes are inserted.
    pub(super) nodes: Vec<Option<Node>>,
    /// Highest-level node used to enter the graph, when populated.
    pub(super) entry: Option<EntryPoint>,
}

/// Report whether `level` should replace the graph's current entry level.
fn should_promote_entry(current: Option<EntryPoint>, level: usize) -> bool {
    level > current.map_or(0, |entry| entry.level)
}

impl Graph {
    /// Allocate an empty graph with fixed node capacity and parameters.
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

    /// Return the current entry point, if a node has been inserted.
    pub(crate) const fn entry(&self) -> Option<EntryPoint> {
        self.entry
    }

    /// Returns the allocated slot count, used for sizing auxiliary buffers.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use crate::hnsw::{graph::Graph, params::HnswParams};
    /// let params = HnswParams::new(4, 8).expect("params must be valid");
    /// let graph = Graph::with_capacity(params, 3);
    /// assert_eq!(graph.capacity(), 3);
    /// ```
    #[must_use]
    pub(crate) const fn capacity(&self) -> usize {
        self.nodes.len()
    }

    /// Iterates over all inserted nodes along with their identifiers.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use crate::hnsw::{graph::{Graph, NodeContext}, params::HnswParams};
    /// let params = HnswParams::new(4, 8).expect("params must be valid");
    /// let mut graph = Graph::with_capacity(params, 2);
    /// graph.insert_first(NodeContext { node: 0, level: 0, sequence: 0 }).expect("insert first");
    /// graph.attach_node(NodeContext { node: 1, level: 0, sequence: 1 }).expect("attach second");
    /// let ids: Vec<_> = graph.nodes_iter().map(|(id, _)| id).collect();
    /// assert_eq!(ids, vec![0, 1]);
    /// ```
    pub(crate) fn nodes_iter(&self) -> impl Iterator<Item = (usize, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| node.as_ref().map(|node_ref| (id, node_ref)))
    }

    /// Insert the first node and make it the graph entry point.
    pub(crate) fn insert_first(&mut self, ctx: NodeContext) -> Result<(), HnswError> {
        self.attach_node(ctx)?;
        self.entry = Some(EntryPoint {
            node: ctx.node,
            level: ctx.level,
        });
        Ok(())
    }

    /// Initialise an unoccupied graph slot with its node context.
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

    /// Inserts the first Kani node without constructing formatted production errors.
    #[cfg(kani)]
    pub(crate) fn insert_first_for_kani(&mut self, ctx: NodeContext) -> Result<(), &'static str> {
        self.attach_node_for_kani(ctx)?;
        self.entry = Some(EntryPoint {
            node: ctx.node,
            level: ctx.level,
        });
        Ok(())
    }

    /// Attaches a Kani node without constructing formatted production errors.
    #[cfg(kani)]
    pub(crate) fn attach_node_for_kani(&mut self, ctx: NodeContext) -> Result<(), &'static str> {
        if ctx.level > self.params.max_level() {
            return Err("node level exceeds max_level");
        }
        let Some(slot) = self.nodes.get_mut(ctx.node) else {
            return Err("node is outside pre-allocated capacity");
        };
        if slot.is_some() {
            return Err("node already exists");
        }
        *slot = Some(Node::new(ctx.level, ctx.sequence));
        Ok(())
    }

    #[cfg(kani)]
    /// Expose entry-promotion criteria to Kani proofs.
    pub(crate) fn should_promote_entry_for_kani(current: Option<EntryPoint>, level: usize) -> bool {
        should_promote_entry(current, level)
    }

    /// Promote a node when its layer is above the current entry point.
    pub(crate) fn promote_entry(&mut self, node: usize, level: usize) {
        if should_promote_entry(self.entry, level) {
            self.entry = Some(EntryPoint { node, level });
        }
    }

    /// Return an inserted node by slot identifier.
    pub(crate) fn node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id).and_then(Option::as_ref)
    }

    /// Return mutable access to an inserted node by slot identifier.
    pub(crate) fn node_mut(&mut self, id: usize) -> Option<&mut Node> {
        self.nodes.get_mut(id).and_then(Option::as_mut)
    }

    /// Retrieves the insertion sequence assigned to a node for deterministic
    /// neighbour ordering.
    ///
    /// Returns `None` when the node slot has not been initialized or the
    /// identifier exceeds the allocated capacity.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use crate::hnsw::{
    ///     graph::{Graph, NodeContext},
    ///     params::HnswParams,
    /// };
    ///
    /// let params = HnswParams::new(8, 16).expect("params");
    /// let mut graph = Graph::with_capacity(params, 2);
    /// graph
    ///     .insert_first(NodeContext {
    ///         node: 0,
    ///         level: 0,
    ///         sequence: 7,
    ///     })
    ///     .expect("insert first node");
    /// assert_eq!(graph.node_sequence(0), Some(7));
    /// assert_eq!(graph.node_sequence(1), None);
    /// ```
    pub(crate) fn node_sequence(&self, id: usize) -> Option<u64> {
        self.node(id).map(Node::sequence)
    }

    /// Report whether `node` names an allocated graph slot.
    pub(crate) fn has_slot(&self, node: usize) -> bool {
        self.nodes.get(node).is_some()
    }

    /// Create a planner borrowing this graph.
    #[inline]
    pub(crate) const fn insertion_planner(&self) -> InsertionPlanner<'_> {
        InsertionPlanner::new(self)
    }

    /// Create an executor borrowing this graph mutably.
    #[inline]
    pub(crate) const fn insertion_executor(&mut self) -> InsertionExecutor<'_> {
        InsertionExecutor::new(self)
    }

    /// Create a layer searcher borrowing this graph.
    #[inline]
    pub(crate) const fn searcher(&self) -> LayerSearcher<'_> {
        LayerSearcher::new(self)
    }

    #[cfg(test)]
    /// Return the graph parameters for invariant tests.
    pub(crate) const fn params(&self) -> &HnswParams {
        &self.params
    }
}
