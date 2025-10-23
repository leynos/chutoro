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
pub(super) struct DescentContext {
    pub(crate) query: usize,
    pub(crate) entry: EntryPoint,
    pub(crate) target_level: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LayerPlanContext {
    pub(crate) query: usize,
    pub(crate) current: usize,
    pub(crate) target_level: usize,
    pub(crate) ef: usize,
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

#[derive(Clone, Copy, Debug)]
pub(super) struct ExtendedSearchContext {
    pub(crate) query: usize,
    pub(crate) entry: usize,
    pub(crate) level: usize,
    pub(crate) ef: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NeighbourSearchContext {
    pub(crate) query: usize,
    pub(crate) level: usize,
    pub(crate) current_dist: f32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ProcessNodeContext {
    pub(crate) query: usize,
    pub(crate) level: usize,
    pub(crate) ef: usize,
    pub(crate) node_id: usize,
}

#[derive(Clone, Debug)]
pub(super) struct ScoredCandidates {
    items: Vec<(usize, f32)>,
}

impl ScoredCandidates {
    pub(super) fn new(candidates: Vec<usize>, distances: Vec<f32>) -> Self {
        debug_assert_eq!(
            candidates.len(),
            distances.len(),
            "candidate and distance batches must align",
        );
        let items = candidates.into_iter().zip(distances).collect();
        Self { items }
    }
}

impl IntoIterator for ScoredCandidates {
    type Item = (usize, f32);
    type IntoIter = std::vec::IntoIter<(usize, f32)>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Graph {
    nodes: Vec<Option<Node>>,
    entry: Option<EntryPoint>,
}

impl Graph {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
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
