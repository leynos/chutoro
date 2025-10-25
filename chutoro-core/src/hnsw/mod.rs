//! CPU implementation of the Hierarchical Navigable Small World (HNSW) graph.
//!
//! The implementation mirrors the concurrency model described in the design
//! documents: Rayon drives parallel insertion, every search acquires a shared
//! lock on the graph, and write access is limited to the mutation window when
//! inserting a node.

mod error;
mod graph;
mod insert;
mod node;
mod params;
mod search;
mod types;
mod validate;

pub use self::{
    error::{HnswError, HnswErrorCode},
    params::HnswParams,
    types::Neighbour,
};

use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use rand::{Rng, SeedableRng, distributions::Standard, rngs::SmallRng};
use rayon::{current_num_threads, current_thread_index, prelude::*};

use crate::DataSource;

use self::{
    graph::{ApplyContext, Graph, NodeContext, SearchContext},
    insert::{TrimJob, TrimResult},
    validate::{validate_batch_distances, validate_distance},
};

/// Parallel CPU HNSW index coordinating insertions through two-phase locking.
#[derive(Debug)]
pub struct CpuHnsw {
    params: HnswParams,
    graph: Arc<RwLock<Graph>>,
    rng: Mutex<SmallRng>,
    worker_rngs: Vec<Mutex<SmallRng>>,
    len: AtomicUsize,
}

impl CpuHnsw {
    /// Builds a new HNSW index from the provided [`DataSource`].
    ///
    /// The first item seeds the entry point. Remaining items are inserted in
    /// parallel with Rayon workers.
    ///
    /// # Errors
    /// Returns [`HnswError::EmptyBuild`] when the data source is empty.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    ///
    /// struct Dummy(Vec<f32>);
    ///
    /// impl DataSource for Dummy {
    ///     fn len(&self) -> usize { self.0.len() }
    ///     fn name(&self) -> &str { "dummy" }
    ///     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    ///         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
    ///         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
    ///         Ok((a - b).abs())
    ///     }
    /// }
    ///
    /// let params = HnswParams::new(2, 4).expect("params must be valid");
    /// let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 3.0]), params)
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
        let node_ctx = NodeContext { node: 0, level };
        validate_distance(source, node_ctx.node, node_ctx.node)?;
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
    pub fn with_capacity(params: HnswParams, capacity: usize) -> Result<Self, HnswError> {
        if capacity == 0 {
            return Err(HnswError::InvalidParameters {
                reason: "capacity must be greater than zero".into(),
            });
        }
        let base_seed = params.rng_seed();
        let worker_rngs = (0..current_num_threads())
            .map(|idx| {
                let seed = base_seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                Mutex::new(SmallRng::seed_from_u64(seed))
            })
            .collect();

        let graph = Graph::with_capacity(params.clone(), capacity);

        Ok(Self {
            rng: Mutex::new(SmallRng::seed_from_u64(base_seed)),
            worker_rngs,
            graph: Arc::new(RwLock::new(graph)),
            len: AtomicUsize::new(0),
            params,
        })
    }

    /// Inserts a node into the graph, performing search under a shared lock.
    pub fn insert<D: DataSource + Sync>(&self, node: usize, source: &D) -> Result<(), HnswError> {
        let level = self.sample_level();
        let node_ctx = NodeContext { node, level };
        if self.try_insert_initial(node_ctx, source)? {
            self.len.store(1, Ordering::Relaxed);
            return Ok(());
        }

        let plan = self.read_graph(|graph| {
            graph
                .insertion_planner()
                .plan(node_ctx, &self.params, source)
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

    fn insert_initial(&self, graph: &mut Graph, ctx: NodeContext) -> Result<(), HnswError> {
        graph.insert_first(ctx.node, ctx.level)
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

        validate_distance(source, ctx.node, ctx.node)?;
        self.write_graph(|graph| {
            if graph.entry().is_none() {
                self.insert_initial(graph, ctx)?;
                Ok(true)
            } else {
                Ok(false)
            }
        })
    }

    fn score_trim_jobs<D: DataSource + Sync>(
        &self,
        trim_jobs: Vec<TrimJob>,
        source: &D,
    ) -> Result<Vec<TrimResult>, HnswError> {
        if trim_jobs.is_empty() {
            return Ok(Vec::new());
        }

        trim_jobs
            .into_par_iter()
            .map(|job| {
                validate_batch_distances(source, job.node, &job.candidates).map(|distances| {
                    let mut scored: Vec<_> = job.candidates.into_iter().zip(distances).collect();
                    // Stable sort preserves input order on ties, favouring the new node
                    // positioned earlier by `prioritise_new_node`.
                    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                    scored.truncate(job.ctx.max_connections);
                    TrimResult {
                        node: job.node,
                        ctx: job.ctx,
                        neighbours: scored.into_iter().map(|(node, _)| node).collect(),
                    }
                })
            })
            .collect::<Result<Vec<_>, HnswError>>()
    }

    /// Searches the index for the `ef` closest neighbours of `query`.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams};
    ///
    /// struct Dummy(Vec<f32>);
    ///
    /// impl DataSource for Dummy {
    ///     fn len(&self) -> usize { self.0.len() }
    ///     fn name(&self) -> &str { "dummy" }
    ///     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    ///         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
    ///         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
    ///         Ok((a - b).abs())
    ///     }
    /// }
    ///
    /// let params = HnswParams::new(2, 4).expect("params must be valid");
    /// let index = CpuHnsw::build(&Dummy(vec![0.0, 1.0, 3.0]), params)
    ///     .expect("build must succeed");
    /// let neighbours = index
    ///     .search(
    ///         &Dummy(vec![0.0, 1.0, 3.0]),
    ///         0,
    ///         std::num::NonZeroUsize::new(4).expect("ef must be non-zero"),
    ///     )
    ///     .expect("search must succeed");
    /// assert!(!neighbours.is_empty());
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
                source,
                SearchContext {
                    query,
                    entry: current,
                    level,
                },
            )?;
        }
        searcher.search_layer(
            source,
            SearchContext {
                query,
                entry: current,
                level: 0,
            }
            .with_ef(ef.get()),
        )
    }

    /// Returns the number of nodes that have been inserted.
    ///
    /// Relaxed ordering is sufficient: inserts publish monotonically
    /// increasing counts and callers only require eventual consistency for
    /// metrics.
    #[must_use]
    #[rustfmt::skip]
    pub fn len(&self) -> usize { self.len.load(Ordering::Relaxed) }

    /// Returns whether the index currently stores no nodes.
    #[must_use]
    #[rustfmt::skip]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

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

#[cfg(test)]
mod tests;
