//! Internal graph representation for the CPU HNSW implementation.

use super::{
    error::HnswError,
    node::Node,
    params::HnswParams,
    types::{EntryPoint, InsertionPlan},
};

#[derive(Clone, Copy, Debug)]
pub(super) struct NodeContext {
    pub(crate) node: usize,
    pub(crate) level: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EdgeContext {
    pub(crate) level: usize,
    pub(crate) max_connections: usize,
}

#[derive(Clone, Copy, Debug)]
struct InsertBaseContext {
    query: usize,
    target_level: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DescentContext {
    base: InsertBaseContext,
    pub(crate) entry: EntryPoint,
}

impl DescentContext {
    pub(crate) fn new(query: usize, entry: EntryPoint, target_level: usize) -> Self {
        Self {
            base: InsertBaseContext {
                query,
                target_level,
            },
            entry,
        }
    }

    pub(crate) fn query(&self) -> usize {
        self.base.query
    }

    pub(crate) fn target_level(&self) -> usize {
        self.base.target_level
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LayerPlanContext {
    base: InsertBaseContext,
    pub(crate) current: usize,
    pub(crate) ef: usize,
}

impl LayerPlanContext {
    pub(crate) fn new(query: usize, current: usize, target_level: usize, ef: usize) -> Self {
        Self {
            base: InsertBaseContext {
                query,
                target_level,
            },
            current,
            ef,
        }
    }

    pub(crate) fn query(&self) -> usize {
        self.base.query
    }

    pub(crate) fn target_level(&self) -> usize {
        self.base.target_level
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
    pub(crate) fn with_ef(self, ef: usize) -> ExtendedSearchContext {
        ExtendedSearchContext { base: self, ef }
    }

    pub(crate) fn with_distance(self, current_dist: f32) -> NeighbourSearchContext {
        NeighbourSearchContext {
            base: self,
            current_dist,
        }
    }

    pub(crate) fn query(&self) -> usize {
        self.query
    }

    pub(crate) fn entry(&self) -> usize {
        self.entry
    }

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
    pub(crate) fn query(&self) -> usize {
        self.base.query()
    }

    pub(crate) fn entry(&self) -> usize {
        self.base.entry()
    }

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
    pub(crate) fn query(&self) -> usize {
        self.base.query()
    }

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
    pub(crate) fn with_capacity(params: HnswParams, capacity: usize) -> Self {
        Self {
            params,
            nodes: vec![None; capacity],
            entry: None,
        }
    }

    pub(crate) fn entry(&self) -> Option<EntryPoint> {
        self.entry
    }

    pub(crate) fn insert_first(&mut self, node: usize, level: usize) -> Result<(), HnswError> {
        self.attach_node(node, level)?;
        self.entry = Some(EntryPoint { node, level });
        Ok(())
    }

    pub(crate) fn attach_node(&mut self, node: usize, level: usize) -> Result<(), HnswError> {
        if level > self.params.max_level() {
            return Err(HnswError::InvalidParameters {
                reason: format!(
                    "level {level} exceeds max_level {}",
                    self.params.max_level()
                ),
            });
        }
        let slot = self
            .nodes
            .get_mut(node)
            .ok_or_else(|| HnswError::InvalidParameters {
                reason: format!("node {node} is outside pre-allocated capacity"),
            })?;
        if slot.is_some() {
            return Err(HnswError::DuplicateNode { node });
        }
        *slot = Some(Node::new(level));
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

    pub(super) fn has_slot(&self, node: usize) -> bool {
        self.nodes.get(node).is_some()
    }
}
