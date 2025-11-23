//! Test-only helpers for exercising and reconfiguring the CPU HNSW.

use std::sync::Mutex;

use rand::{SeedableRng, rngs::SmallRng};

use crate::hnsw::{
    distance_cache::DistanceCache, error::HnswError, graph::Graph, params::HnswParams,
};

use super::{CpuHnsw, rng::build_worker_rngs};

impl CpuHnsw {
    /// Test-only healing hook that re-enforces reachability and bidirectionality.
    ///
    /// Compiled only for tests to avoid production overhead; intended to stabilise
    /// property-based mutation checks that rely on post-commit healing passes.
    pub fn heal_for_test(&self) {
        self.write_graph(|graph| {
            let mut executor = graph.insertion_executor();
            executor.heal_reachability(self.params.max_connections());
            executor.enforce_bidirectional_all(self.params.max_connections());
        });
    }

    pub(crate) fn inspect_graph<R>(&self, f: impl FnOnce(&Graph) -> R) -> R {
        self.read_graph(f)
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
        self.write_graph(|graph| graph.set_params(&self.params));
    }
}
