//! Test-only helpers for exercising and reconfiguring the CPU HNSW.

use std::sync::Mutex;

use rand::{SeedableRng, rngs::SmallRng};

use crate::hnsw::{
    distance_cache::DistanceCache, error::HnswError, graph::Graph, params::HnswParams,
};

use super::{CpuHnsw, internal, rng::build_worker_rngs};

impl CpuHnsw {
    /// Test-only healing hook that re-enforces reachability and bidirectionality.
    ///
    /// Compiled only for tests to avoid production overhead; intended to stabilize
    /// property-based mutation checks that rely on post-commit healing passes.
    ///
    /// # Panics
    ///
    /// Panics when the test graph lock cannot be acquired, or localized
    /// reciprocity validation fails for a touched adjacency list.
    pub fn heal_for_test(&self) {
        let max_connections = self.params.max_connections();
        let healed = self.write_graph(|graph| {
            {
                let mut executor = graph.insertion_executor();
                executor.heal_reachability(max_connections);
            }
            let touched = graph.take_touched_nodes();
            {
                let mut executor = graph.insertion_executor();
                executor.enforce_bidirectional_for_touched(&touched, max_connections);
            }
            Ok(())
        });
        if let Err(err) = healed {
            panic!("graph lock during heal_for_test: {err}");
        }
    }

    /// Runs `f` against the shared graph, reporting a poisoned lock instead of
    /// panicking.
    ///
    /// The helper is a fixture, so it returns the lock failure rather than
    /// unwinding through it: a `#[test]` caller adds `.expect(...)` at its own
    /// boundary, and the poisoned-lock regression can inspect the error value
    /// and prove nothing else changed.
    ///
    /// # Errors
    ///
    /// Returns [`HnswError::LockPoisoned`] with `resource: "graph"` when the
    /// graph lock is already poisoned.
    pub(crate) fn inspect_graph<R>(&self, f: impl FnOnce(&Graph) -> R) -> Result<R, HnswError> {
        self.read_graph(|graph| Ok(f(graph)))
    }

    /// Reports whether this thread is currently marked as holding the graph write lock.
    pub(crate) fn current_thread_holds_write_graph_for_test() -> bool {
        internal::current_thread_holds_write_graph()
    }

    /// Enables the write-lock marker until the returned guard is dropped.
    pub(crate) fn enable_write_graph_marker_for_test() -> WriteGraphMarkerGuard {
        internal::enable_write_graph_marker();
        WriteGraphMarkerGuard
    }

    /// Deletes a graph node and updates the index length when deletion succeeds.
    pub(crate) fn delete_node_for_test(&mut self, node: usize) -> Result<bool, HnswError> {
        let deleted = self.write_graph(|graph| graph.delete_node(node))?;
        if deleted {
            self.len.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(deleted)
    }

    /// Swaps in `params` for the graph and every index structure derived from
    /// it.
    ///
    /// The graph lock is acquired before anything else changes, so a poisoned
    /// lock leaves the index exactly as it was: the parameters, random
    /// generators, and distance cache are only replaced once the graph has
    /// accepted the new parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HnswError::LockPoisoned`] with `resource: "graph"` when the
    /// graph lock is poisoned; no index state changes in that case.
    pub(crate) fn reconfigure_for_test(&mut self, params: HnswParams) -> Result<(), HnswError> {
        self.write_graph(|graph| {
            graph.set_params(&params);
            Ok(())
        })?;
        let base_seed = params.rng_seed();
        self.rng = Mutex::new(SmallRng::seed_from_u64(base_seed));
        self.worker_rngs = build_worker_rngs(base_seed);
        self.distance_cache = DistanceCache::new(*params.distance_cache_config());
        self.params = params;
        Ok(())
    }
}

/// Restores the write-graph marker when the enabling test scope ends.
pub(crate) struct WriteGraphMarkerGuard;

impl Drop for WriteGraphMarkerGuard {
    fn drop(&mut self) {
        internal::disable_write_graph_marker();
    }
}

#[cfg(test)]
mod tests {
    //! Regression coverage for the fallible graph helpers.
    //!
    //! These cases deliberately poison the graph lock, which means they panic
    //! inside `catch_unwind`. The opt-in Cranelift development route cannot
    //! unwind that panic at all, so `dev-test` excludes this exact case and the
    //! stable test gate is the one that runs it.

    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    /// Poisons the graph lock by panicking while its write guard is alive.
    fn poison_graph_lock(index: &CpuHnsw) {
        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            // The binding must outlive the panic: unwinding drops the guard
            // while the thread is panicking, which is what poisons the lock.
            let _guard = index
                .graph
                .write()
                .expect("fresh graph lock must be available");
            panic!("poison the graph lock for this test");
        }));
        assert!(poisoned.is_err(), "the poisoning panic must have unwound");
    }

    #[test]
    fn graph_helpers_report_poisoning_without_reconfiguring() {
        let params = HnswParams::new(2, 4).expect("initial parameters must be valid");
        let mut index = CpuHnsw::with_capacity(params.clone(), 3).expect("index must initialize");

        poison_graph_lock(&index);

        assert!(
            matches!(
                index.inspect_graph(|_| ()),
                Err(HnswError::LockPoisoned { resource: "graph" })
            ),
            "inspection must report the poisoned graph lock",
        );

        let replacement = HnswParams::new(3, 6).expect("replacement parameters must be valid");
        assert!(
            matches!(
                index.reconfigure_for_test(replacement),
                Err(HnswError::LockPoisoned { resource: "graph" })
            ),
            "reconfiguration must report the poisoned graph lock",
        );
        assert_eq!(
            index.params, params,
            "failed reconfiguration must retain the active parameters",
        );
    }
}
