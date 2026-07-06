//! Private helpers for graph access, initial insertion, and sequence handling.

use std::sync::{RwLockReadGuard, RwLockWriteGuard, atomic::Ordering};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
use std::sync::atomic::AtomicUsize;

use crate::{
    DataSource,
    hnsw::{
        error::HnswError,
        graph::{Graph, NodeContext},
        validate::validate_distance,
    },
};

use super::CpuHnsw;

#[cfg(test)]
thread_local! {
    // `current_thread_holds_write_graph` and `WriteGraphScope` use TLS depth
    // so only the thread executing the write closure reports ownership. A
    // process-wide boolean would make one Rayon worker look like every worker
    // holds the write lock, creating cross-thread false positives and hiding
    // same-thread false negatives.
    static WRITE_GRAPH_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
// `enable_write_graph_marker`/`disable_write_graph_marker` are process-wide so
// any Rayon worker entering `write_graph` is tracked once a test opts in.
// Collapsing this enable flag and the TLS depth into one global boolean would
// lose either cross-worker opt-in or same-thread ownership accuracy, creating
// cross-thread false positives or false negatives.
static WRITE_GRAPH_MARKER_ENABLE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn enable_write_graph_marker() {
    WRITE_GRAPH_MARKER_ENABLE_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn disable_write_graph_marker() {
    let previous = WRITE_GRAPH_MARKER_ENABLE_COUNT.fetch_sub(1, Ordering::Relaxed);
    assert!(previous > 0, "write graph marker disable must match enable");
}

#[cfg(test)]
pub(super) fn current_thread_holds_write_graph() -> bool {
    WRITE_GRAPH_DEPTH.with(|depth| depth.get() > 0)
}

#[cfg(test)]
struct WriteGraphScope;

#[cfg(test)]
impl WriteGraphScope {
    fn enter_if_enabled() -> Option<Self> {
        if WRITE_GRAPH_MARKER_ENABLE_COUNT.load(Ordering::Relaxed) == 0 {
            return None;
        }
        WRITE_GRAPH_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Some(Self)
    }
}

#[cfg(test)]
impl Drop for WriteGraphScope {
    fn drop(&mut self) {
        WRITE_GRAPH_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

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
        #[cfg(test)]
        let _write_graph_scope = WriteGraphScope::enter_if_enabled();
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
