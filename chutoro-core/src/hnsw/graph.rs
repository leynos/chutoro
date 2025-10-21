//! Internal graph representation for the CPU HNSW implementation.

use super::{error::HnswError, node::Node, types::EntryPoint};

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
}
