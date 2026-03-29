//! CPU-resident HNSW implementation that drives insertion, search, and
//! invariant checking for the chutoro graph. Coordinates Rayon workers,
//! sharded RNGs, and the shared distance cache while exposing the public
//! `CpuHnsw` API used by the CLI and tests.

mod build;
pub(super) mod internal;
pub(super) mod rng;
pub(super) mod trim;

#[cfg(test)]
pub(super) mod test_helpers;

use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use rand::{SeedableRng, rngs::SmallRng};

use crate::DataSource;

use super::{
    distance_cache::DistanceCache,
    error::HnswError,
    graph::{Graph, SearchContext},
    helpers::{EnsureQueryArgs, ensure_query_present, normalize_neighbour_order},
    invariants::HnswInvariantChecker,
    params::HnswParams,
    types::Neighbour,
};

use self::rng::build_worker_rngs;

/// Parallel CPU HNSW index coordinating insertions through two-phase locking.
#[derive(Debug)]
pub struct CpuHnsw {
    pub(super) params: HnswParams,
    pub(super) graph: Arc<RwLock<Graph>>,
    rng: Mutex<SmallRng>,
    worker_rngs: Vec<Mutex<SmallRng>>,
    distance_cache: DistanceCache,
    insert_mutex: Mutex<()>,
    next_sequence: AtomicU64,
    len: AtomicUsize,
}

impl CpuHnsw {
    /// Creates an empty index with the desired capacity.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use chutoro_core::{CpuHnsw, HnswParams};
    /// let params = HnswParams::new(2, 8).expect("params");
    /// let index = CpuHnsw::with_capacity(params, 16).expect("capacity must be > 0");
    /// assert!(index.is_empty());
    /// ```
    pub fn with_capacity(params: HnswParams, capacity: usize) -> Result<Self, HnswError> {
        if capacity == 0 {
            return Err(HnswError::InvalidParameters {
                reason: "capacity must be greater than zero".into(),
            });
        }
        let base_seed = params.rng_seed();
        let worker_rngs = build_worker_rngs(base_seed);

        let cache = DistanceCache::new(*params.distance_cache_config());
        let graph = Graph::with_capacity(params.clone(), capacity);

        Ok(Self {
            rng: Mutex::new(SmallRng::seed_from_u64(base_seed)),
            worker_rngs,
            graph: Arc::new(RwLock::new(graph)),
            distance_cache: cache,
            insert_mutex: Mutex::new(()),
            next_sequence: AtomicU64::new(0),
            len: AtomicUsize::new(0),
            params,
        })
    }

    /// Searches the index for the `ef` closest neighbours of `query`.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
    /// #         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
    /// #         Ok((a - b).abs())
    /// #     }
    /// # }
    /// let params = HnswParams::new(2, 4).expect("params");
    /// let data = Dummy(vec![0.0, 1.0, 3.5]);
    /// let index = CpuHnsw::build(&data, params).expect("build must succeed");
    /// let neighbours = index
    ///     .search(&data, 0, std::num::NonZeroUsize::new(2).unwrap())
    ///     .expect("search must succeed");
    /// assert_eq!(neighbours[0].id, 0);
    /// ```
    pub fn search<D: DataSource + Sync>(
        &self,
        source: &D,
        query: usize,
        ef: NonZeroUsize,
    ) -> Result<Vec<Neighbour>, HnswError> {
        let graph = self.read_graph_guard()?;
        let entry = graph.entry().ok_or(HnswError::GraphEmpty)?;
        let searcher = graph.searcher();
        let mut current = entry.node;
        for level in (1..=entry.level).rev() {
            current = searcher.greedy_search_layer(
                Some(&self.distance_cache),
                source,
                SearchContext {
                    query,
                    entry: current,
                    level,
                },
            )?;
        }
        let mut neighbours = searcher.search_layer(
            Some(&self.distance_cache),
            source,
            SearchContext {
                query,
                entry: current,
                level: 0,
            }
            .with_ef(ef.get()),
        )?;
        normalize_neighbour_order(&mut neighbours);
        ensure_query_present(
            &self.distance_cache,
            EnsureQueryArgs {
                source,
                query,
                ef,
                neighbours: &mut neighbours,
            },
        )?;
        Ok(neighbours)
    }

    /// Returns the number of nodes that have been inserted.
    #[must_use]
    #[rustfmt::skip]
    pub fn len(&self) -> usize { self.len.load(Ordering::Relaxed) }

    /// Returns whether the index currently stores no nodes.
    #[must_use]
    #[rustfmt::skip]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Returns a handle for checking structural invariants.
    #[must_use]
    pub fn invariants(&self) -> HnswInvariantChecker<'_> {
        HnswInvariantChecker::new(self)
    }
}

#[cfg(test)]
mod tests;
