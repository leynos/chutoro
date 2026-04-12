//! CPU-backed session types for incremental clustering workflows.
//!
//! This module introduces the initial public session scaffolding used by the
//! roadmap's incremental clustering work. In this phase the session is an
//! inspectable, empty shell that owns configuration, an allocated HNSW index,
//! and the backing data source without yet performing append or refresh work.

use std::{num::NonZeroUsize, sync::Arc};

use crate::{CandidateEdge, ChutoroError, CpuHnsw, DataSource, HnswParams, MstEdge, Result};

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

/// A live clustering session prepared for later incremental operations.
///
/// The initial implementation allocates an empty HNSW index and records the
/// configuration and backing source, but does not seed existing source items
/// into the session. Follow-up roadmap items add append, refresh, bootstrap,
/// and label-snapshot behaviour.
///
/// # Examples
/// ```rust,no_run
/// use std::sync::Arc;
///
/// use chutoro_core::{
///     ChutoroBuilder, ClusteringSession, DataSource, DataSourceError, MetricDescriptor,
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
/// let source = Arc::new(Dummy(vec![0.0, 1.0]));
/// let session: ClusteringSession<Dummy> = ChutoroBuilder::new()
///     .build_session(source)
///     .expect("session creation must succeed");
///
/// assert_eq!(session.point_count(), 0);
/// assert_eq!(session.snapshot_version(), 0);
/// ```
#[derive(Debug)]
pub struct ClusteringSession<D: DataSource + Sync> {
    config: SessionConfig,
    index: CpuHnsw,
    _core_distances: Vec<f32>,
    _mst_edges: Vec<MstEdge>,
    _historical_edges: Vec<CandidateEdge>,
    _pending_edges: Vec<CandidateEdge>,
    _labels: Arc<Vec<usize>>,
    snapshot_version: u64,
    _source: Arc<D>,
    _last_refresh_len: usize,
}

impl<D: DataSource + Sync> ClusteringSession<D> {
    pub(crate) fn new(config: SessionConfig, source: Arc<D>) -> Result<Self> {
        let index = CpuHnsw::with_capacity(config.hnsw_params().clone(), source.len().max(1))
            .map_err(|error| ChutoroError::CpuHnswFailure {
                code: Arc::from(error.code().as_str()),
                message: Arc::from(error.to_string()),
            })?;

        Ok(Self {
            config,
            index,
            _core_distances: Vec::new(),
            _mst_edges: Vec::new(),
            _historical_edges: Vec::new(),
            _pending_edges: Vec::new(),
            _labels: Arc::new(Vec::new()),
            snapshot_version: 0,
            _source: source,
            _last_refresh_len: 0,
        })
    }

    /// Returns the validated configuration used by the session.
    #[must_use]
    pub fn config(&self) -> &SessionConfig {
        &self.config
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
