//! CPU implementation of the Hierarchical Navigable Small World (HNSW) graph.
//!
//! The implementation mirrors the concurrency model described in the design
//! documents: Rayon drives parallel insertion, every search acquires a shared
//! lock on the graph, and write access is limited to the mutation window when
//! inserting a node.

mod error;
mod graph;
mod params;

pub use self::{
    error::{HnswError, HnswErrorCode},
    graph::Neighbour,
    params::HnswParams,
};

use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use rand::{Rng, SeedableRng, distributions::Standard, rngs::SmallRng};
use rayon::prelude::*;

use crate::DataSource;

use self::graph::{ApplyContext, ExtendedSearchContext, Graph, NodeContext, SearchContext};

/// Parallel CPU HNSW index coordinating insertions through two-phase locking.
#[derive(Debug)]
pub struct CpuHnsw {
    params: HnswParams,
    graph: Arc<RwLock<Graph>>,
    rng: Mutex<SmallRng>,
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
        index.insert_initial(0, source)?;
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
        Ok(Self {
            rng: Mutex::new(SmallRng::seed_from_u64(params.rng_seed())),
            graph: Arc::new(RwLock::new(Graph::with_capacity(capacity))),
            len: AtomicUsize::new(0),
            params,
        })
    }

    /// Inserts a node into the graph, performing search under a shared lock.
    pub fn insert<D: DataSource + Sync>(&self, node: usize, source: &D) -> Result<(), HnswError> {
        let level = self.sample_level();
        {
            let mut graph = self.graph.write().expect("graph lock poisoned");
            if graph.entry().is_none() {
                graph.insert_first(node, level)?;
                graph::validate_distance(source, node, node)?;
                self.len.store(1, Ordering::Relaxed);
                return Ok(());
            }
        }
        let plan = {
            // Phase 1: Plan insertion under read lock
            let graph = self.graph.read().expect("graph lock poisoned");
            graph.plan_insertion(NodeContext { node, level }, &self.params, source)
        }?;
        {
            // Phase 2: Apply insertion under write lock
            let mut graph = self.graph.write().expect("graph lock poisoned");
            graph.apply_insertion(ApplyContext {
                node: NodeContext { node, level },
                params: &self.params,
                plan,
                source,
            })?;
        }
        self.len.fetch_add(1, Ordering::Relaxed);
        Ok(())
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
        let mut current = entry.node;
        for level in (1..=entry.level).rev() {
            current = graph.greedy_search_layer(
                source,
                SearchContext {
                    query,
                    entry: current,
                    level,
                },
            )?;
        }
        graph.search_layer(
            source,
            ExtendedSearchContext {
                query,
                entry: current,
                level: 0,
                ef: ef.get(),
            },
        )
    }

    /// Returns the number of nodes that have been inserted.
    ///
    /// Relaxed ordering is sufficient: inserts publish monotonically
    /// increasing counts and callers only require eventual consistency for
    /// metrics.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Returns whether the index currently stores no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn insert_initial<D: DataSource + Sync>(
        &self,
        node: usize,
        source: &D,
    ) -> Result<(), HnswError> {
        let level = self.sample_level();
        {
            let mut graph = self.graph.write().expect("graph lock poisoned");
            graph.insert_first(node, level)?;
        }
        graph::validate_distance(source, node, node)?;
        self.len.store(1, Ordering::Relaxed);
        Ok(())
    }

    fn sample_level(&self) -> usize {
        let mut rng = self.rng.lock().expect("rng mutex poisoned");
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
