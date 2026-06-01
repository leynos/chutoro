//! Session configuration types for incremental clustering workflows.
//!
//! This module owns the small value types carried by
//! [`super::ClusteringSession`]. Keeping refresh policy and validated session
//! parameters here lets `session/mod.rs` expose the public API while the
//! construction and append implementation consume one coherent
//! [`SessionConfig`] value.

use std::num::NonZeroUsize;

use crate::HnswParams;

/// Refresh behaviour for a [`super::ClusteringSession`].
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

/// Validated configuration carried by a [`super::ClusteringSession`].
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
