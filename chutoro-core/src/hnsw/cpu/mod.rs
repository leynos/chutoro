//! CPU-resident HNSW implementation that drives insertion, search, and
//! invariant checking for the chutoro graph. Coordinates Rayon workers,
//! sharded RNGs, and the shared distance cache while exposing the public
//! `CpuHnsw` API used by the CLI and tests.

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
use rayon::prelude::*;

use crate::DataSource;

use super::{
    distance_cache::DistanceCache,
    error::HnswError,
    graph::{ApplyContext, Graph, NodeContext, SearchContext},
    helpers::{EnsureQueryArgs, ensure_query_present, normalise_neighbour_order},
    insert::PlanningInputs,
    invariants::HnswInvariantChecker,
    params::HnswParams,
    types::Neighbour,
    validate::validate_distance,
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
    /// Builds a new HNSW index from the provided [`DataSource`].
    ///
    /// The first item seeds the entry point; remaining items are inserted in
    /// parallel with Rayon workers.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use crate::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    ///
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
    /// let params = HnswParams::new(2, 4).expect("params must be valid");
    /// let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 2.0]), params)
    ///     .expect("build must succeed");
    /// assert_eq!(index.len(), 3);
    /// ```
    pub fn build<D: DataSource + Sync>(source: &D, params: HnswParams) -> Result<Self, HnswError> {
        let items = source.len();
        if items == 0 {
            return Err(HnswError::EmptyBuild);
        }
        let index = Self::with_capacity(params, items)?;
        let level = index.sample_level()?;
        let sequence = index.allocate_sequence();
        let node_ctx = NodeContext {
            node: 0,
            level,
            sequence,
        };
        validate_distance(
            Some(&index.distance_cache),
            source,
            node_ctx.node,
            node_ctx.node,
        )?;
        {
            index.write_graph(|graph| index.insert_initial(graph, node_ctx))?;
        }
        index.len.store(1, Ordering::Relaxed);
        if items > 1 {
            (1..items)
                .into_par_iter()
                .try_for_each(|node| index.insert(node, source))?;
        }
        Ok(index)
    }

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

    /// Inserts a node into the graph, performing search under a shared lock.
    ///
    /// # Examples
    /// ```rust,ignore
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
    /// let data = Dummy(vec![0.0, 2.0, 4.0]);
    /// let index = CpuHnsw::build(&data, params).expect("build must succeed");
    /// index.insert(1, &data).expect("insert must succeed");
    /// ```
    pub fn insert<D: DataSource + Sync>(&self, node: usize, source: &D) -> Result<(), HnswError> {
        let _insertion_guard = self
            .insert_mutex
            .lock()
            .map_err(|_| HnswError::LockPoisoned {
                resource: "insert mutex",
            })?;
        let level = self.sample_level()?;
        let sequence = self.allocate_sequence();
        let node_ctx = NodeContext {
            node,
            level,
            sequence,
        };
        if self.try_insert_initial(node_ctx, source)? {
            self.len.store(1, Ordering::Relaxed);
            return Ok(());
        }

        let cache = &self.distance_cache;
        let plan = self.read_graph(|graph| {
            graph.insertion_planner().plan(PlanningInputs {
                ctx: node_ctx,
                params: &self.params,
                source,
                cache: Some(cache),
            })
        })?;
        let (prepared, trim_jobs) = self.write_graph(|graph| {
            let mut executor = graph.insertion_executor();
            executor.apply(
                node_ctx,
                ApplyContext {
                    params: &self.params,
                    plan,
                },
            )
        })?;
        let trim_results = self.score_trim_jobs(trim_jobs, source)?;
        self.write_graph(|graph| {
            let mut executor = graph.insertion_executor();
            executor.commit(prepared, trim_results)
        })?;
        self.len.fetch_add(1, Ordering::Relaxed);
        Ok(())
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
        normalise_neighbour_order(&mut neighbours);
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
mod tests {
    use super::*;
    use crate::{
        MetricDescriptor, datasource::DataSource, error::DataSourceError, hnsw::HnswParams,
    };
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering as AtomicOrdering},
        },
        thread,
        time::Duration,
    };

    #[test]
    fn insert_waits_for_mutex() {
        let params = HnswParams::new(2, 4).expect("params").with_rng_seed(31);
        let index = Arc::new(CpuHnsw::with_capacity(params, 2).expect("index"));
        let source = Arc::new(TestSource::new(vec![0.0, 1.0]));

        let guard = index.insert_mutex.lock().expect("mutex");
        let started = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));

        let handle = {
            let index = Arc::clone(&index);
            let source = Arc::clone(&source);
            let started = Arc::clone(&started);
            let finished = Arc::clone(&finished);
            thread::spawn(move || {
                started.store(true, AtomicOrdering::SeqCst);
                index.insert(0, &*source).expect("insert must succeed");
                finished.store(true, AtomicOrdering::SeqCst);
            })
        };

        thread::sleep(Duration::from_millis(50));
        assert!(started.load(AtomicOrdering::SeqCst));
        assert!(
            !finished.load(AtomicOrdering::SeqCst),
            "insert should block while the mutex is held"
        );

        drop(guard);
        handle.join().expect("thread joins");
        assert!(finished.load(AtomicOrdering::SeqCst));
    }

    #[derive(Clone)]
    struct TestSource {
        data: Vec<f32>,
    }

    impl TestSource {
        fn new(data: Vec<f32>) -> Self {
            Self { data }
        }
    }

    impl DataSource for TestSource {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn name(&self) -> &str {
            "test"
        }

        fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
            Ok((self.data[left] - self.data[right]).abs())
        }

        fn metric_descriptor(&self) -> MetricDescriptor {
            MetricDescriptor::new("test")
        }
    }
}
