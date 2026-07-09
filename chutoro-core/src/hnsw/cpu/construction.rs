//! Constructors and bulk-build entry points for [`CpuHnsw`].

use super::*;

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

        // Non-harvesting path: use try_for_each to avoid edge allocation
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

        // Node 0 has no edges to harvest (it's the entry point with no prior nodes)
        let edges = if items > 1 {
            EdgeHarvest::from_parallel_inserts(&index, source, items)?
        } else {
            EdgeHarvest::default()
        };

        Ok((index, edges))
    }

    /// Shared initial setup for both `build` and `build_with_edges`.
    ///
    /// Creates the index, inserts the entry point (node 0), and returns
    /// the index ready for parallel insertion of remaining nodes.
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
}
