//! CPU-backed session types for incremental clustering workflows.
//!
//! This module provides the public session surface used by the roadmap's
//! incremental clustering work. The session owns configuration, an allocated
//! HNSW index, and the backing data source. It currently supports append-only
//! insertion and buffers harvested candidate edges for later refresh work.

use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    CandidateEdge, ChutoroError, CpuHnsw, DataSource, DataSourceError, HnswError, HnswParams,
    MetricDescriptor, MstEdge, Result,
};
use tracing::{debug, instrument, warn};

/// Refresh behaviour for a [`ClusteringSession`].
///
/// The initial session API supports only manual refresh or a simple
/// item-count threshold for later roadmap work. Drift triggers and baseline
/// policies remain deferred to later milestones.
///
/// # Examples
/// ```
/// use std::num::NonZeroUsize;
///
/// use chutoro_core::SessionRefreshPolicy;
///
/// let policy = SessionRefreshPolicy::manual()
///     .with_refresh_every_n(NonZeroUsize::new(32));
/// assert_eq!(policy.refresh_every_n().map(NonZeroUsize::get), Some(32));
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SessionRefreshPolicy {
    refresh_every_n: Option<NonZeroUsize>,
}

impl SessionRefreshPolicy {
    /// Creates a policy that leaves refreshes under explicit caller control.
    #[must_use]
    pub fn manual() -> Self {
        Self::default()
    }

    /// Sets the optional append threshold that should trigger a refresh later.
    #[must_use]
    pub fn with_refresh_every_n(mut self, refresh_every_n: Option<NonZeroUsize>) -> Self {
        self.refresh_every_n = refresh_every_n;
        self
    }

    /// Returns the configured append threshold for automatic refresh.
    #[must_use]
    pub fn refresh_every_n(&self) -> Option<NonZeroUsize> {
        self.refresh_every_n
    }
}

/// Validated configuration carried by a [`ClusteringSession`].
///
/// Instances are derived from [`crate::ChutoroBuilder`] so the session stores
/// strong, already-validated clustering parameters.
///
/// # Examples
/// ```rust,no_run
/// use std::{num::NonZeroUsize, sync::Arc};
///
/// use chutoro_core::{
///     ChutoroBuilder, ClusteringSession, DataSource, DataSourceError,
///     MetricDescriptor, SessionRefreshPolicy,
/// };
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
///     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("abs") }
/// }
///
/// let source = Arc::new(Dummy(vec![1.0, 2.0, 4.0]));
/// let session: ClusteringSession<Dummy> = ChutoroBuilder::new()
///     .with_min_cluster_size(3)
///     .with_session_refresh_policy(
///         SessionRefreshPolicy::manual()
///             .with_refresh_every_n(NonZeroUsize::new(8)),
///     )
///     .build_session(source)
///     .expect("session configuration must be valid");
///
/// assert_eq!(session.config().min_cluster_size().get(), 3);
/// assert_eq!(
///     session.config().refresh_policy().refresh_every_n().map(NonZeroUsize::get),
///     Some(8)
/// );
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SessionConfig {
    min_cluster_size: NonZeroUsize,
    hnsw_params: HnswParams,
    refresh_policy: SessionRefreshPolicy,
}

impl SessionConfig {
    pub(crate) fn new(
        min_cluster_size: NonZeroUsize,
        hnsw_params: HnswParams,
        refresh_policy: SessionRefreshPolicy,
    ) -> Self {
        Self {
            min_cluster_size,
            hnsw_params,
            refresh_policy,
        }
    }

    /// Returns the minimum cluster size carried into the session.
    #[must_use]
    pub fn min_cluster_size(&self) -> NonZeroUsize {
        self.min_cluster_size
    }

    /// Returns the HNSW parameters used for the session index.
    #[must_use]
    pub fn hnsw_params(&self) -> &HnswParams {
        &self.hnsw_params
    }

    /// Returns the session refresh policy.
    #[must_use]
    pub fn refresh_policy(&self) -> &SessionRefreshPolicy {
        &self.refresh_policy
    }
}

/// A live clustering session for append-oriented incremental clustering.
///
/// The current implementation allocates an empty HNSW index and records the
/// configuration and backing source. Calls to [`Self::append`] insert point
/// indices through the public HNSW edge-harvesting path and retain harvested
/// candidate edges for later refresh work. Follow-up roadmap items add refresh,
/// bootstrap, and label-snapshot behaviour.
///
/// ## Concurrency model
///
/// `append` requires `&mut self`, giving the caller exclusive access while the
/// session mutates the live index and pending edge buffer. Read-only accessors
/// remain available through shared references. `_labels` uses `Arc<Vec<usize>>`,
/// so later label snapshots can be shared without blocking the writer.
/// Concurrent read-only access through a shared `RwLock` guard is verified by the
/// `concurrent_readers_observe_consistent_point_count` and
/// `snapshot_version_is_immutable_under_concurrent_readers` tests in `session/tests.rs`.
///
/// ## `pending_edges` memory usage
///
/// `pending_edges` accumulates [`CandidateEdge`] values as points are
/// inserted. Each `CandidateEdge` has a raw field payload of
/// `2 × size_of::<usize>() + 4` bytes (two endpoint indices and one `f32`
/// distance). The struct is then rounded up to its 8-byte alignment, which
/// makes it 24 bytes on 64-bit targets.
/// In the worst case a session that has inserted *N* points
/// will hold at most *N* × *M* edges, where *M* is the HNSW `max_connections`
/// parameter (default 16). For 10 000 points with `M = 16` that is ~3.84 MB —
/// modest for a transient buffer, but callers must be aware that
/// `pending_edges` grows without bound until a future `refresh()` call
/// (roadmap item 11.1.4) drains it. Long-lived sessions inserting very many
/// points should plan for a periodic refresh cadence or monitor the per-point
/// harvest volume via the `chutoro.session.harvested_edges` counter (enabled
/// with the `metrics` Cargo feature).
///
/// # Examples
/// ```rust,no_run
/// use std::sync::Arc;
///
/// use chutoro_core::{
///     ChutoroBuilder, ChutoroError, ClusteringSession, DataSource,
///     DataSourceError, MetricDescriptor,
/// };
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
///     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("abs") }
/// }
///
/// # fn main() -> Result<(), ChutoroError> {
/// let source = Arc::new(Dummy(vec![0.0, 1.0]));
/// let mut session: ClusteringSession<Dummy> = ChutoroBuilder::new()
///     .build_session(source)?;
///
/// session.append(&[0, 1])?;
///
/// assert_eq!(session.point_count(), 2);
/// assert_eq!(session.snapshot_version(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ClusteringSession<D: DataSource + Send + Sync> {
    config: SessionConfig,
    index: CpuHnsw,
    _core_distances: Vec<f32>,
    _mst_edges: Vec<MstEdge>,
    _historical_edges: Vec<CandidateEdge>,
    pending_edges: Vec<CandidateEdge>,
    _labels: Arc<Vec<usize>>,
    snapshot_version: u64,
    source: Arc<D>,
    _last_refresh_len: usize,
}

// Verify ClusteringSession is Send + Sync when its DataSource is Send + Sync.
// _DummySrc is Send by default (no non-Send fields); the compiler enforces the
// bound at the call site of assert_send_sync.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}

    struct _DummySrc;

    impl DataSource for _DummySrc {
        fn len(&self) -> usize {
            0
        }

        fn name(&self) -> &str {
            "_dummy"
        }

        fn distance(&self, _i: usize, _j: usize) -> std::result::Result<f32, DataSourceError> {
            Ok(0.0)
        }

        fn metric_descriptor(&self) -> MetricDescriptor {
            MetricDescriptor::new("_dummy")
        }
    }

    assert_send_sync::<ClusteringSession<_DummySrc>>();
};

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    fn append_index_error(&self, index: usize) -> ChutoroError {
        ChutoroError::DataSource {
            data_source: Arc::from(self.source.name()),
            error: DataSourceError::OutOfBounds { index },
        }
    }

    fn map_hnsw_error(&self, error: HnswError) -> ChutoroError {
        crate::cpu_pipeline::map_cpu_hnsw_error(self.source.as_ref(), error)
    }

    fn new_with_index_result(
        config: SessionConfig,
        source: Arc<D>,
        index: std::result::Result<CpuHnsw, HnswError>,
    ) -> Result<Self> {
        let index = index.map_err(|error| {
            let code = Arc::from(error.code().as_str());
            let message = Arc::from(error.to_string());
            warn!(
                code = ?code,
                message = %message,
                "CpuHnsw index allocation failed; returning CpuHnswFailure"
            );
            ChutoroError::CpuHnswFailure { code, message }
        })?;
        debug!(
            min_cluster_size = %config.min_cluster_size(),
            "ClusteringSession allocated: empty HNSW index ready"
        );

        #[cfg(feature = "metrics")]
        {
            metrics::describe_counter!(
                "chutoro.session.append.errors_total",
                "Total number of append failures, labelled by reason."
            );
            metrics::describe_histogram!(
                "chutoro.session.append.point_seconds",
                metrics::Unit::Seconds,
                "Per-point HNSW insertion latency in seconds."
            );
            metrics::describe_counter!(
                "chutoro.session.harvested_edges",
                metrics::Unit::Count,
                "Total harvested candidate edges buffered for refresh."
            );
        }

        Ok(Self {
            config,
            index,
            _core_distances: Vec::new(),
            _mst_edges: Vec::new(),
            _historical_edges: Vec::new(),
            pending_edges: Vec::new(),
            _labels: Arc::new(Vec::new()),
            snapshot_version: 0,
            source,
            _last_refresh_len: 0,
        })
    }

    fn new_with_capacity(config: SessionConfig, source: Arc<D>, capacity: usize) -> Result<Self> {
        let index = CpuHnsw::with_capacity(config.hnsw_params().clone(), capacity);
        Self::new_with_index_result(config, source, index)
    }

    pub(crate) fn new(config: SessionConfig, source: Arc<D>) -> Result<Self> {
        let capacity = source.len().max(1);
        Self::new_with_capacity(config, source, capacity)
    }

    #[cfg(test)]
    pub(crate) fn new_failing_for_test(config: SessionConfig, source: Arc<D>) -> Result<Self> {
        Self::new_with_index_result(
            config,
            source,
            Err(HnswError::InvalidParameters {
                reason: "test-injected HNSW construction failure".to_owned(),
            }),
        )
    }

    /// Returns the validated configuration used by the session.
    #[must_use]
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Appends new point indices to the live HNSW index.
    ///
    /// Each index is inserted with [`CpuHnsw::insert_harvesting`], and every
    /// harvested [`CandidateEdge`] is retained in the session's pending edge
    /// buffer for later refresh work. The method is fail-fast: if an insertion
    /// fails, earlier successful insertions and their pending edges remain in
    /// the session.
    ///
    /// # Errors
    ///
    /// Returns [`ChutoroError::DataSource`] when the backing source rejects an
    /// index or distance query. Returns [`ChutoroError::CpuHnswFailure`] for
    /// duplicate indices, non-finite distances, lock poisoning, or graph
    /// invariant failures reported by the HNSW index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use chutoro_core::{
    /// #     ChutoroBuilder, ChutoroError, DataSource, DataSourceError,
    /// #     MetricDescriptor,
    /// # };
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
    /// #         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
    /// #         Ok((a - b).abs())
    /// #     }
    /// #     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("abs") }
    /// # }
    /// # fn main() -> Result<(), ChutoroError> {
    /// let source = Arc::new(Dummy(vec![0.0, 1.0]));
    /// let mut session = ChutoroBuilder::new().build_session(source)?;
    ///
    /// session.append(&[0, 1])?;
    /// assert_eq!(session.point_count(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self, indices), fields(count = indices.len()), level = "debug")]
    pub fn append(&mut self, indices: &[usize]) -> Result<()> {
        for &index in indices {
            if index >= self.source.len() {
                warn!(
                    index,
                    source_len = self.source.len(),
                    "append rejected: index out of bounds"
                );
                #[cfg(feature = "metrics")]
                metrics::counter!(
                    "chutoro.session.append.errors_total",
                    "reason" => "out_of_bounds"
                )
                .increment(1);
                return Err(self.append_index_error(index));
            }

            #[cfg(feature = "metrics")]
            let t0 = std::time::Instant::now();

            let edges = self
                .index
                .insert_harvesting(index, self.source.as_ref())
                .map_err(|error| {
                    let chutoro_error = self.map_hnsw_error(error);
                    warn!(
                        index,
                        error = ?chutoro_error,
                        "append rejected: HNSW insertion failed"
                    );
                    #[cfg(feature = "metrics")]
                    metrics::counter!(
                        "chutoro.session.append.errors_total",
                        "reason" => "hnsw_failure"
                    )
                    .increment(1);
                    chutoro_error
                })?;

            #[cfg(feature = "metrics")]
            metrics::histogram!("chutoro.session.append.point_seconds")
                .record(t0.elapsed().as_secs_f64());

            let harvested = edges.len();
            self.pending_edges.extend(edges);

            #[cfg(feature = "metrics")]
            metrics::counter!("chutoro.session.harvested_edges").increment(harvested as u64);

            debug!(
                index,
                harvested, "append: point inserted and edges buffered"
            );
        }
        Ok(())
    }

    /// Returns the number of points currently inserted into the session.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.index.len()
    }

    /// Returns the most recent published snapshot version.
    #[must_use]
    pub fn snapshot_version(&self) -> u64 {
        self.snapshot_version
    }
}

#[cfg(test)]
mod tests;
