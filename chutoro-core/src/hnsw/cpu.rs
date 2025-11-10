//! CPU-resident HNSW implementation that drives insertion, search, and
//! invariant checking for the chutoro graph. Coordinates Rayon workers,
//! sharded RNGs, and the shared distance cache while exposing the public
//! `CpuHnsw` API used by the CLI and tests.
use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use rand::{Rng, SeedableRng, distributions::Standard, rngs::SmallRng};
use rayon::{current_num_threads, current_thread_index, prelude::*};

use crate::DataSource;

use super::{
    distance_cache::DistanceCache,
    error::HnswError,
    graph::{ApplyContext, Graph, NodeContext, SearchContext},
    insert::{PlanningInputs, TrimJob, TrimResult},
    invariants::HnswInvariantChecker,
    params::HnswParams,
    types::{Neighbour, RankedNeighbour},
    validate::{validate_batch_distances, validate_distance},
};

/// Mixes worker RNG seeds by the 64-bit golden ratio to decorrelate threads.
const WORKER_SEED_SPACING: u64 = 0x9E37_79B9_7F4A_7C15;

/// Parallel CPU HNSW index coordinating insertions through two-phase locking.
#[derive(Debug)]
pub struct CpuHnsw {
    pub(super) params: HnswParams,
    pub(super) graph: Arc<RwLock<Graph>>,
    rng: Mutex<SmallRng>,
    worker_rngs: Vec<Mutex<SmallRng>>,
    distance_cache: DistanceCache,
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
    /// ```
    /// use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
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
        let level = index.sample_level();
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
            let mut graph = index.graph.write().expect("graph lock poisoned");
            index.insert_initial(&mut graph, node_ctx)?;
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
    /// ```
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
        let worker_rngs = (0..current_num_threads())
            .map(|idx| {
                // Mix the worker index into the base seed to keep per-thread RNGs
                // decorrelated when Rayon reuses threads across searches.
                let seed = base_seed ^ ((idx as u64 + 1).wrapping_mul(WORKER_SEED_SPACING));
                Mutex::new(SmallRng::seed_from_u64(seed))
            })
            .collect();

        let cache = DistanceCache::new(*params.distance_cache_config());
        let graph = Graph::with_capacity(params.clone(), capacity);

        Ok(Self {
            rng: Mutex::new(SmallRng::seed_from_u64(base_seed)),
            worker_rngs,
            graph: Arc::new(RwLock::new(graph)),
            distance_cache: cache,
            next_sequence: AtomicU64::new(0),
            len: AtomicUsize::new(0),
            params,
        })
    }

    /// Inserts a node into the graph, performing search under a shared lock.
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
    /// let data = Dummy(vec![0.0, 2.0, 4.0]);
    /// let index = CpuHnsw::build(&data, params).expect("build must succeed");
    /// index.insert(1, &data).expect("insert must succeed");
    /// ```
    pub fn insert<D: DataSource + Sync>(&self, node: usize, source: &D) -> Result<(), HnswError> {
        let level = self.sample_level();
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
        let graph = self.graph.read().expect("graph lock poisoned");
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

    fn insert_initial(&self, graph: &mut Graph, ctx: NodeContext) -> Result<(), HnswError> {
        graph.insert_first(ctx)
    }

    fn read_graph<R>(&self, f: impl FnOnce(&Graph) -> R) -> R {
        let guard = self.graph.read().expect("graph lock poisoned");
        f(&guard)
    }

    fn write_graph<R>(&self, f: impl FnOnce(&mut Graph) -> R) -> R {
        let mut guard = self.graph.write().expect("graph lock poisoned");
        f(&mut guard)
    }

    #[cfg(test)]
    pub(crate) fn inspect_graph<R>(&self, f: impl FnOnce(&Graph) -> R) -> R {
        self.read_graph(f)
    }

    fn try_insert_initial<D: DataSource + Sync>(
        &self,
        ctx: NodeContext,
        source: &D,
    ) -> Result<bool, HnswError> {
        if self.read_graph(|graph| graph.entry().is_some()) {
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

    /// Scores trim jobs in parallel, validating candidate, sequence, and
    /// distance lengths before emitting ranked neighbour lists capped at each
    /// edge context's `max_connections`.
    ///
    /// The caller supplies trimmed candidates gathered while the graph lock is
    /// held. This method then validates the batched distances without the lock
    /// and deterministically orders neighbours by distance and insertion
    /// sequence so ties remain stable.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use chutoro_core::{
    ///     CpuHnsw,
    ///     DataSource,
    ///     DataSourceError,
    ///     HnswParams,
    ///     hnsw::graph::EdgeContext,
    ///     hnsw::insert::executor::TrimJob,
    /// };
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
    /// let params = HnswParams::new(1, 2).unwrap();
    /// let hnsw = CpuHnsw::with_capacity(params, 2).unwrap();
    /// let trim_jobs = vec![TrimJob {
    ///     node: 0,
    ///     ctx: EdgeContext { level: 0, max_connections: 1 },
    ///     candidates: vec![0],
    ///     sequences: vec![0],
    /// }];
    /// let results = hnsw.score_trim_jobs(trim_jobs, &Dummy(vec![0.0])).unwrap();
    /// assert_eq!(results[0].neighbours, vec![0]);
    /// ```
    pub(crate) fn score_trim_jobs<D: DataSource + Sync>(
        &self,
        trim_jobs: Vec<TrimJob>,
        source: &D,
    ) -> Result<Vec<TrimResult>, HnswError> {
        if trim_jobs.is_empty() {
            return Ok(Vec::new());
        }

        trim_jobs
            .into_par_iter()
            .map(|job| -> Result<TrimResult, HnswError> {
                let TrimJob {
                    node,
                    ctx,
                    candidates,
                    sequences,
                } = job;

                if candidates.len() != sequences.len() {
                    return Err(HnswError::InvalidParameters {
                        reason: format!(
                            "trim job candidates ({}) must match sequence count ({})",
                            candidates.len(),
                            sequences.len()
                        ),
                    });
                }

                let distances = validate_batch_distances(
                    Some(&self.distance_cache),
                    source,
                    node,
                    &candidates,
                )?;
                if distances.len() != candidates.len() {
                    return Err(HnswError::InvalidParameters {
                        reason: format!(
                            "trim job distance count ({}) mismatches candidates ({})",
                            distances.len(),
                            candidates.len()
                        ),
                    });
                }

                let mut scored = Vec::with_capacity(candidates.len());
                for (index, id) in candidates.into_iter().enumerate() {
                    let sequence = sequences[index];
                    let distance = distances[index];
                    scored.push(RankedNeighbour::new(id, distance, sequence));
                }
                scored.sort_unstable();
                scored.truncate(ctx.max_connections);
                Ok(TrimResult {
                    node,
                    ctx,
                    neighbours: scored
                        .into_iter()
                        .map(|neighbour| neighbour.into_neighbour().id)
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, HnswError>>()
    }

    fn allocate_sequence(&self) -> u64 {
        self.next_sequence.fetch_add(1, Ordering::Relaxed)
    }

    fn sample_level(&self) -> usize {
        if let Some(index) = current_thread_index() {
            if let Some(rng) = self.worker_rngs.get(index) {
                let mut guard = rng.lock().expect("worker rng mutex poisoned");
                return self.sample_level_from_rng(&mut guard);
            }
        }

        let mut rng = self.rng.lock().expect("rng mutex poisoned");
        self.sample_level_from_rng(&mut rng)
    }

    fn sample_level_from_rng(&self, rng: &mut SmallRng) -> usize {
        let mut level = 0_usize;
        while level < self.params.max_level() {
            let draw: f64 = rng.sample(Standard);
            if self.params.should_stop(draw) {
                break;
            }
            level += 1;
        }
        level
    }
}

fn normalise_neighbour_order(neighbours: &mut [Neighbour]) {
    neighbours.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then_with(|| left.id.cmp(&right.id))
    });
}
