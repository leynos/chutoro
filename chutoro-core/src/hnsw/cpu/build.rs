//! Build and insertion orchestration for the CPU HNSW index.
//!
//! Keeps the top-level `cpu` module focused on the steady-state search API
//! while this module owns graph construction, candidate-edge harvesting, and
//! insertion coordination.

use std::sync::atomic::Ordering;

use rayon::prelude::*;

use crate::{
    DataSource,
    hnsw::{
        error::HnswError,
        graph::{ApplyContext, NodeContext},
        insert::{PlanningInputs, extract_candidate_edges},
        params::HnswParams,
        types::{CandidateEdge, EdgeHarvest},
        validate::validate_distance,
    },
};

use super::CpuHnsw;

impl EdgeHarvest {
    /// Collects edges from parallel insertions using Rayon `map` → `try_reduce`.
    ///
    /// Inserts nodes `1..items` in parallel, accumulating discovered edges into
    /// a single sorted harvest. The returned edges are sorted by insertion
    /// sequence for deterministic ordering.
    pub(super) fn from_parallel_inserts<D: DataSource + Sync>(
        index: &CpuHnsw,
        source: &D,
        items: usize,
    ) -> Result<Self, HnswError> {
        let edges = (1..items)
            .into_par_iter()
            .map(|node| index.insert_with_edges(node, source))
            .try_reduce(Vec::new, |mut acc, node_edges| {
                acc.extend(node_edges);
                Ok(acc)
            })?;

        Ok(Self::from_unsorted(edges))
    }
}

impl CpuHnsw {
    /// Builds a new HNSW index from the provided [`DataSource`].
    ///
    /// The first item seeds the entry point; remaining items are inserted in
    /// parallel with Rayon workers. This method uses a non-harvesting path
    /// that avoids edge allocation overhead.
    ///
    /// Use [`Self::build_with_edges`] if you need candidate edges for MST
    /// construction.
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
        let index = Self::build_initial(source, params)?;
        let items = source.len();

        // Non-harvesting path: use try_for_each to avoid edge allocation.
        if items > 1 {
            (1..items)
                .into_par_iter()
                .try_for_each(|node| index.insert(node, source))?;
        }

        Ok(index)
    }

    /// Builds an HNSW index and returns candidate edges for MST construction.
    ///
    /// The first item seeds the entry point (no edges harvested for it).
    /// Remaining items are inserted in parallel using Rayon, with edges
    /// accumulated via `map` → `reduce` into a global edge list.
    ///
    /// The returned edges are sorted by insertion sequence for deterministic
    /// ordering, then by the natural `Ord` (distance, source, target, sequence).
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{CpuHnsw, DataSource, DataSourceError, HnswParams, MetricDescriptor};
    ///
    /// struct Dummy(Vec<f32>);
    /// impl DataSource for Dummy {
    ///     fn len(&self) -> usize { self.0.len() }
    ///     fn name(&self) -> &str { "dummy" }
    ///     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    ///         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
    ///         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
    ///         Ok((a - b).abs())
    ///     }
    ///     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("test") }
    /// }
    ///
    /// let params = HnswParams::new(2, 4).expect("params");
    /// let (index, edges) = CpuHnsw::build_with_edges(&Dummy(vec![0.0, 1.0, 2.0]), params)
    ///     .expect("build");
    ///
    /// assert_eq!(index.len(), 3);
    /// // Edges connect nodes discovered during insertion
    /// assert!(edges.iter().all(|e| e.source() < 3 && e.target() < 3));
    /// ```
    pub fn build_with_edges<D: DataSource + Sync>(
        source: &D,
        params: HnswParams,
    ) -> Result<(Self, EdgeHarvest), HnswError> {
        let index = Self::build_initial(source, params)?;
        let items = source.len();

        // Node 0 has no edges to harvest because it seeds the entry point.
        let edges = if items > 1 {
            EdgeHarvest::from_parallel_inserts(&index, source, items)?
        } else {
            EdgeHarvest::default()
        };

        Ok((index, edges))
    }

    /// Shared initial setup for both `build` and `build_with_edges`.
    ///
    /// Creates the index, inserts the entry point (node 0), and returns the
    /// index ready for parallel insertion of remaining nodes.
    fn build_initial<D: DataSource + Sync>(
        source: &D,
        params: HnswParams,
    ) -> Result<Self, HnswError> {
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
        index.write_graph(|graph| index.insert_initial(graph, node_ctx))?;
        index.len.store(1, Ordering::Relaxed);

        Ok(index)
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
        self.insert_internal(node, source, None)
    }

    /// Inserts a node and returns the candidate edges discovered during planning.
    ///
    /// This is used internally by [`EdgeHarvest::from_parallel_inserts`] to
    /// accumulate edges via Rayon `map` → `reduce` for MST construction.
    fn insert_with_edges<D: DataSource + Sync>(
        &self,
        node: usize,
        source: &D,
    ) -> Result<Vec<CandidateEdge>, HnswError> {
        let mut edges = Vec::new();
        self.insert_internal(node, source, Some(&mut edges))?;
        Ok(edges)
    }

    /// Core insertion logic with an optional edge sink for MST harvesting.
    fn insert_internal<D: DataSource + Sync>(
        &self,
        node: usize,
        source: &D,
        edge_sink: Option<&mut Vec<CandidateEdge>>,
    ) -> Result<(), HnswError> {
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

        if let Some(edges) = edge_sink {
            edges.extend(extract_candidate_edges(node, sequence, &plan));
        }

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
}
