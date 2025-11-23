//! Private helpers for graph access, initial insertion, and sequence handling.

use std::sync::{RwLockReadGuard, RwLockWriteGuard, atomic::Ordering};

use crate::{
    DataSource,
    hnsw::{
        error::HnswError,
        graph::{Graph, NodeContext},
        validate::validate_distance,
    },
};

use super::CpuHnsw;

impl CpuHnsw {
    pub(super) fn insert_initial(
        &self,
        graph: &mut Graph,
        ctx: NodeContext,
    ) -> Result<(), HnswError> {
        graph.insert_first(ctx)
    }

    pub(super) fn read_graph_guard(&self) -> Result<RwLockReadGuard<'_, Graph>, HnswError> {
        self.graph
            .read()
            .map_err(|_| HnswError::LockPoisoned { resource: "graph" })
    }

    pub(super) fn write_graph_guard(&self) -> Result<RwLockWriteGuard<'_, Graph>, HnswError> {
        self.graph
            .write()
            .map_err(|_| HnswError::LockPoisoned { resource: "graph" })
    }

    pub(super) fn read_graph<R>(
        &self,
        f: impl FnOnce(&Graph) -> Result<R, HnswError>,
    ) -> Result<R, HnswError> {
        let guard = self.read_graph_guard()?;
        f(&guard)
    }

    pub(crate) fn write_graph<R>(
        &self,
        f: impl FnOnce(&mut Graph) -> Result<R, HnswError>,
    ) -> Result<R, HnswError> {
        let mut guard = self.write_graph_guard()?;
        f(&mut guard)
    }

    pub(super) fn allocate_sequence(&self) -> u64 {
        self.next_sequence.fetch_add(1, Ordering::Relaxed)
    }

    pub(super) fn try_insert_initial<D: DataSource + Sync>(
        &self,
        ctx: NodeContext,
        source: &D,
    ) -> Result<bool, HnswError> {
        if self.read_graph(|graph| Ok(graph.entry().is_some()))? {
            return Ok(false);
        }

        validate_distance(Some(&self.distance_cache), source, ctx.node, ctx.node)?;
        self.write_graph(|graph| {
            if graph.entry().is_none() {
                self.insert_initial(graph, ctx)?;
                Ok(true)
            } else {
                Ok(false)
            }
        })
    }
}
