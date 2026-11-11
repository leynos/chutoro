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
    //! The poisoning case deliberately panics inside `catch_unwind`. The opt-in
    //! Cranelift development route cannot unwind that panic at all, so
    //! `dev-test` excludes that exact case and the stable test gate is the one
    //! that runs it. The successful case stays in the development route: it
    //! proves the replacement is applied, which is what makes the poisoning
    //! case's preservation assertions mean something.

    use std::num::NonZeroUsize;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::time::Duration;

    use rand::RngCore as _;

    use super::*;
    use crate::hnsw::cpu::rng::mix_worker_seed;

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

    /// Advances both generators and caches one distance, returning copies of
    /// the generators as they now stand.
    ///
    /// A preserved generator is only meaningful if it had already progressed:
    /// comparing two never-drawn generators would also pass if the failed
    /// reconfiguration had rebuilt them from the seed. The returned copies are
    /// the expected continuation, so the check after the failure is a real
    /// state comparison rather than a second independent draw.
    fn advance_generators_and_cache(index: &mut CpuHnsw) -> (SmallRng, SmallRng) {
        index
            .rng
            .lock()
            .expect("fresh generator lock must be available")
            .next_u64();
        index
            .worker_rngs
            .first()
            .expect("a built index holds worker generators")
            .lock()
            .expect("fresh worker generator lock must be available")
            .next_u64();

        let metric = crate::MetricDescriptor::new("test-helpers");
        let miss = match index.distance_cache.begin_lookup(&metric, 0, 1) {
            crate::hnsw::distance_cache::LookupOutcome::Miss(miss) => miss,
            crate::hnsw::distance_cache::LookupOutcome::Hit(_) => {
                panic!("a fresh cache must not already hold this distance")
            }
        };
        index
            .distance_cache
            .complete_miss(miss, 1.5)
            .expect("the distance is finite");

        let fallback = index.rng.lock().expect("generator lock").clone();
        let worker = index
            .worker_rngs
            .first()
            .expect("a built index holds worker generators")
            .lock()
            .expect("worker generator lock")
            .clone();
        (fallback, worker)
    }

    /// Returns a parameter set whose every field differs from `params`.
    ///
    /// `HnswParams::new` keeps the default seed and cache configuration, so a
    /// replacement built from it alone would leave those fields equal and the
    /// preservation assertion vacuous for them. Deserialized parameters are
    /// the real source of mixed values, so every field is varied here.
    fn distinguishable(params: &HnswParams) -> HnswParams {
        let cache_entries =
            NonZeroUsize::new(params.distance_cache_config().max_entries().get() + 7)
                .expect("a cache size larger than the current one is non-zero");
        HnswParams::new(params.max_connections() + 1, params.ef_construction() + 2)
            .expect("a wider replacement must be valid")
            .with_max_level(params.max_level() + 1)
            .with_rng_seed(params.rng_seed().wrapping_add(1))
            .with_distance_cache_max_entries(cache_entries)
            .with_distance_cache_ttl(Some(Duration::from_secs(7)))
    }

    #[test]
    fn graph_helpers_report_poisoning_without_reconfiguring() {
        let params = HnswParams::new(2, 4).expect("initial parameters must be valid");
        let mut index = CpuHnsw::with_capacity(params.clone(), 3).expect("index must initialize");
        let (mut expected_fallback, mut expected_worker) = advance_generators_and_cache(&mut index);

        poison_graph_lock(&index);

        assert!(
            matches!(
                index.inspect_graph(|_| ()),
                Err(HnswError::LockPoisoned { resource: "graph" })
            ),
            "inspection must report the poisoned graph lock",
        );

        let replacement = distinguishable(&params);
        assert_ne!(
            replacement, params,
            "the replacement must differ in every field, or this proves nothing",
        );
        assert!(
            matches!(
                index.reconfigure_for_test(replacement),
                Err(HnswError::LockPoisoned { resource: "graph" })
            ),
            "reconfiguration must report the poisoned graph lock",
        );
        // Whole-struct equality covers the seed and the cache configuration
        // as well as the two widths.
        assert_eq!(
            index.params, params,
            "failed reconfiguration must retain the active parameters",
        );
        // The cache kept its distance, and the generators resume from exactly
        // where the advance left them: these are the draws that would have
        // followed it, which a generator rebuilt from `params` would not
        // reproduce. `reconfigure_for_test` would have replaced all three.
        let metric = crate::MetricDescriptor::new("test-helpers");
        assert!(
            matches!(
                index.distance_cache.begin_lookup(&metric, 0, 1),
                crate::hnsw::distance_cache::LookupOutcome::Hit(_)
            ),
            "failed reconfiguration must retain the cached distance",
        );
        assert_eq!(
            index.rng.lock().expect("generator lock").next_u64(),
            expected_fallback.next_u64(),
            "failed reconfiguration must not rebuild the fallback generator",
        );
        assert_eq!(
            index
                .worker_rngs
                .first()
                .expect("a built index holds worker generators")
                .lock()
                .expect("worker generator lock")
                .next_u64(),
            expected_worker.next_u64(),
            "failed reconfiguration must not rebuild the worker generators",
        );
    }

    /// Replaces every derived structure of a healthy index.
    ///
    /// The poisoning case asserts that a refused swap changes nothing. That
    /// assertion is only meaningful if a *permitted* swap changes everything,
    /// which is what this case measures: without it, a `reconfigure_for_test`
    /// that acquired the lock and then returned early would satisfy the
    /// regression while silently leaving `make dev-test` testing stale state.
    ///
    /// The graph is checked through its own parameter copy as well as the
    /// index field, because the two are set at different points of the helper
    /// and a partial swap could update one without the other.
    #[test]
    fn reconfigure_replaces_parameters_generators_and_cache() {
        let params = HnswParams::new(2, 4).expect("initial parameters must be valid");
        let mut index = CpuHnsw::with_capacity(params.clone(), 3).expect("index must initialize");
        let (mut expected_fallback, mut expected_worker) = advance_generators_and_cache(&mut index);

        let replacement = distinguishable(&params);
        index
            .reconfigure_for_test(replacement.clone())
            .expect("an unpoisoned index must accept new parameters");

        assert_eq!(
            index.params, replacement,
            "reconfiguration must adopt the new parameters",
        );
        index
            .inspect_graph(|graph| {
                assert_eq!(
                    graph.params(),
                    &replacement,
                    "reconfiguration must hand the new parameters to the graph",
                );
            })
            .expect("graph inspection must succeed");

        // The generators now continue from the replacement seed rather than
        // from where the advance left them. One draw from each live generator
        // is compared both ways, so the assertion needs no assumption about how
        // far the replacement sequences have advanced and consumes exactly the
        // state it inspects.
        let mut replacement_fallback = SmallRng::seed_from_u64(replacement.rng_seed());
        let mut replacement_worker =
            SmallRng::seed_from_u64(mix_worker_seed(replacement.rng_seed(), 0));
        let fallback_draw = index.rng.lock().expect("generator lock").next_u64();
        assert_eq!(
            fallback_draw,
            replacement_fallback.next_u64(),
            "the fallback generator must resume from the replacement seed",
        );
        assert_ne!(
            fallback_draw,
            expected_fallback.next_u64(),
            "reconfiguration must replace the fallback generator, not continue it",
        );
        let worker_draw = index
            .worker_rngs
            .first()
            .expect("a built index holds worker generators")
            .lock()
            .expect("worker generator lock")
            .next_u64();
        assert_eq!(
            worker_draw,
            replacement_worker.next_u64(),
            "reconfiguration must rebuild the worker generators from the new seed",
        );
        assert_ne!(
            worker_draw,
            expected_worker.next_u64(),
            "reconfiguration must not leave the worker generators where they were",
        );

        let metric = crate::MetricDescriptor::new("test-helpers");
        assert!(
            matches!(
                index.distance_cache.begin_lookup(&metric, 0, 1),
                crate::hnsw::distance_cache::LookupOutcome::Miss(_)
            ),
            "reconfiguration must replace the distance cache",
        );
    }
}
