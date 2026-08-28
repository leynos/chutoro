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
    /// Panics when the test graph lock cannot be acquired, or when healing
    /// leaves a non-reciprocal edge behind.
    pub fn heal_for_test(&self) {
        let max_connections = self.params.max_connections();
        let healed = self.write_graph(|graph| {
            let mut executor = graph.insertion_executor();
            executor.heal_reachability(max_connections);
            executor.enforce_bidirectional_all(max_connections);
            Ok(executor.find_reciprocity_violation(max_connections))
        });
        match healed {
            Ok(None) => {}
            Ok(Some(violation)) => {
                panic!("heal_for_test left a reciprocity violation: {violation:?}")
            }
            Err(err) => panic!("graph lock during heal_for_test: {err}"),
        }
    }

    pub(crate) fn inspect_graph<R>(&self, f: impl FnOnce(&Graph) -> R) -> R {
        match self.read_graph(|graph| Ok(f(graph))) {
            Ok(result) => result,
            Err(err) => panic!("graph lock during inspect_graph: {err}"),
        }
    }

    pub(crate) fn current_thread_holds_write_graph_for_test() -> bool {
        internal::current_thread_holds_write_graph()
    }

    pub(crate) fn enable_write_graph_marker_for_test() -> WriteGraphMarkerGuard {
        internal::enable_write_graph_marker();
        WriteGraphMarkerGuard
    }

    pub(crate) fn delete_node_for_test(&mut self, node: usize) -> Result<bool, HnswError> {
        let deleted = self.write_graph(|graph| graph.delete_node(node))?;
        if deleted {
            self.len.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(deleted)
    }

    pub(crate) fn reconfigure_for_test(&mut self, params: HnswParams) {
        let base_seed = params.rng_seed();
        self.rng = Mutex::new(SmallRng::seed_from_u64(base_seed));
        self.worker_rngs = build_worker_rngs(base_seed);
        self.distance_cache = DistanceCache::new(*params.distance_cache_config());
        self.params = params;
        let reconfigured = self.write_graph(|graph| {
            graph.set_params(&self.params);
            Ok(())
        });
        if let Err(err) = reconfigured {
            panic!("graph lock during reconfigure_for_test: {err}");
        }
    }
}

pub(crate) struct WriteGraphMarkerGuard;

impl Drop for WriteGraphMarkerGuard {
    fn drop(&mut self) {
        internal::disable_write_graph_marker();
    }
}
