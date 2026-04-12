//! Builder utilities for configuring Chutoro orchestration.
//!
//! Exposes the execution strategy selection surface and builder validation used before constructing [`Chutoro`] instances.

use std::num::NonZeroUsize;

#[cfg(feature = "cpu")]
use std::sync::Arc;

#[cfg(feature = "cpu")]
use crate::{ClusteringSession, DataSource, HnswParams, SessionConfig, SessionRefreshPolicy};
use crate::{Result, chutoro::Chutoro, error::ChutoroError};

/// Indicates how [`Chutoro`] selects a compute backend when [`Chutoro::run`] is
/// invoked.
///
/// `Auto` resolves backends deterministically. Today it runs the CPU backend
/// (enabled by the default `cpu` feature). Once a real GPU backend ships, it
/// will select the GPU implementation when available and fall back to the CPU
/// otherwise so behaviour stays stable across builds.
///
/// # Examples
/// ```
/// use chutoro_core::ExecutionStrategy;
///
/// let strategy = ExecutionStrategy::Auto;
/// assert!(matches!(strategy, ExecutionStrategy::Auto));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Allow the library to select an appropriate backend automatically.
    Auto,
    /// Restrict execution to the CPU implementation.
    CpuOnly,
    /// Prefer a GPU implementation if one is available.
    GpuPreferred,
}

/// Configures and constructs [`Chutoro`] instances.
///
/// # Examples
/// ```
/// use chutoro_core::{ChutoroBuilder, ExecutionStrategy};
///
/// let chutoro = ChutoroBuilder::new()
///     .with_min_cluster_size(8)
///     .with_execution_strategy(ExecutionStrategy::CpuOnly)
///     .build()
///     .expect("builder configuration is valid");
/// assert_eq!(chutoro.min_cluster_size().get(), 8);
/// assert_eq!(chutoro.execution_strategy(), ExecutionStrategy::CpuOnly);
/// ```
#[derive(Debug, Clone)]
pub struct ChutoroBuilder {
    min_cluster_size: usize,
    execution_strategy: ExecutionStrategy,
    max_bytes: Option<u64>,
    #[cfg(feature = "cpu")]
    hnsw_params: HnswParams,
    #[cfg(feature = "cpu")]
    session_refresh_policy: SessionRefreshPolicy,
}

impl Default for ChutoroBuilder {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            execution_strategy: ExecutionStrategy::Auto,
            max_bytes: None,
            #[cfg(feature = "cpu")]
            hnsw_params: HnswParams::default(),
            #[cfg(feature = "cpu")]
            session_refresh_policy: SessionRefreshPolicy::manual(),
        }
    }
}

#[derive(Clone, Copy)]
struct ValidatedBuilderConfig {
    min_cluster_size: NonZeroUsize,
}

impl ChutoroBuilder {
    /// Creates a builder populated with default parameters.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ChutoroBuilder, ExecutionStrategy};
    ///
    /// let builder = ChutoroBuilder::new();
    /// assert_eq!(builder.min_cluster_size(), 5);
    /// assert_eq!(builder.execution_strategy(), ExecutionStrategy::Auto);
    /// ```
    #[rustfmt::skip]
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Overrides the minimum cluster size.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// let builder = ChutoroBuilder::new().with_min_cluster_size(10);
    /// assert_eq!(builder.min_cluster_size(), 10);
    /// ```
    #[must_use]
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Returns the configured minimum cluster size.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// let builder = ChutoroBuilder::new().with_min_cluster_size(3);
    /// assert_eq!(builder.min_cluster_size(), 3);
    /// ```
    #[rustfmt::skip]
    #[must_use]
    pub fn min_cluster_size(&self) -> usize { self.min_cluster_size }

    /// Sets the execution strategy to use when running the algorithm.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ChutoroBuilder, ExecutionStrategy};
    ///
    /// let builder = ChutoroBuilder::new().with_execution_strategy(ExecutionStrategy::CpuOnly);
    /// assert_eq!(builder.execution_strategy(), ExecutionStrategy::CpuOnly);
    /// ```
    #[must_use]
    pub fn with_execution_strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.execution_strategy = strategy;
        self
    }

    /// Returns the currently configured execution strategy.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ChutoroBuilder, ExecutionStrategy};
    ///
    /// let builder = ChutoroBuilder::new().with_execution_strategy(ExecutionStrategy::CpuOnly);
    /// assert_eq!(builder.execution_strategy(), ExecutionStrategy::CpuOnly);
    /// ```
    #[rustfmt::skip]
    #[must_use]
    pub fn execution_strategy(&self) -> ExecutionStrategy { self.execution_strategy }

    /// Sets an upper bound on estimated peak memory (in bytes).
    ///
    /// When set, [`Chutoro::run`] will compute a pre-flight estimate and
    /// return [`ChutoroError::MemoryLimitExceeded`] if the estimate exceeds
    /// this limit.  Omit this call to leave the guard disabled (the default).
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// let builder = ChutoroBuilder::new().with_max_bytes(1_073_741_824);
    /// assert_eq!(builder.max_bytes(), Some(1_073_741_824));
    /// ```
    #[must_use]
    pub fn with_max_bytes(mut self, bytes: u64) -> Self {
        self.max_bytes = Some(bytes);
        self
    }

    /// Returns the configured memory limit, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// assert_eq!(ChutoroBuilder::new().max_bytes(), None);
    /// ```
    #[rustfmt::skip]
    #[must_use]
    pub fn max_bytes(&self) -> Option<u64> { self.max_bytes }

    /// Sets the HNSW parameters used when constructing clustering sessions.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ChutoroBuilder, HnswParams};
    ///
    /// let params = HnswParams::new(8, 32).expect("params must be valid");
    /// let builder = ChutoroBuilder::new().with_hnsw_params(params.clone());
    /// assert_eq!(builder.hnsw_params(), &params);
    /// ```
    #[cfg(feature = "cpu")]
    #[must_use]
    pub fn with_hnsw_params(mut self, params: HnswParams) -> Self {
        self.hnsw_params = params;
        self
    }

    /// Returns the HNSW parameters used for session construction.
    #[cfg(feature = "cpu")]
    #[must_use]
    pub fn hnsw_params(&self) -> &HnswParams {
        &self.hnsw_params
    }

    /// Sets the refresh policy carried into clustering sessions.
    ///
    /// # Examples
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use chutoro_core::{ChutoroBuilder, SessionRefreshPolicy};
    ///
    /// let policy = SessionRefreshPolicy::manual()
    ///     .with_refresh_every_n(NonZeroUsize::new(16));
    /// let builder = ChutoroBuilder::new().with_session_refresh_policy(policy);
    /// assert_eq!(builder.session_refresh_policy(), &policy);
    /// ```
    #[cfg(feature = "cpu")]
    #[must_use]
    pub fn with_session_refresh_policy(mut self, policy: SessionRefreshPolicy) -> Self {
        self.session_refresh_policy = policy;
        self
    }

    /// Returns the refresh policy used for session construction.
    #[cfg(feature = "cpu")]
    #[must_use]
    pub fn session_refresh_policy(&self) -> &SessionRefreshPolicy {
        &self.session_refresh_policy
    }

    /// Validates the configuration and constructs a [`Chutoro`] instance.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// let chutoro = ChutoroBuilder::new().build().expect("configuration is valid");
    /// assert_eq!(chutoro.min_cluster_size().get(), 5);
    /// ```
    pub fn build(self) -> Result<Chutoro> {
        let validated = self.validate_for_batch()?;

        Ok(Chutoro::new(
            validated.min_cluster_size,
            self.execution_strategy,
            self.max_bytes,
        ))
    }

    /// Validates the configuration and constructs an empty
    /// [`ClusteringSession`].
    ///
    /// The returned session owns the provided data source reference but does
    /// not seed existing items into the live index yet. Empty and undersized
    /// data sources are therefore accepted at construction time.
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
    /// let source = Arc::new(Dummy(vec![0.0, 1.0, 2.0]));
    /// let session: ClusteringSession<Dummy> = ChutoroBuilder::new()
    ///     .build_session(source)
    ///     .expect("session configuration must be valid");
    /// assert_eq!(session.point_count(), 0);
    /// ```
    #[cfg(feature = "cpu")]
    pub fn build_session<D: DataSource + Sync>(
        self,
        source: Arc<D>,
    ) -> Result<ClusteringSession<D>> {
        let validated = self.validate_for_session()?;
        let config = SessionConfig::new(
            validated.min_cluster_size,
            self.hnsw_params,
            self.session_refresh_policy,
        );

        ClusteringSession::new(config, source)
    }

    fn validate_min_cluster_size(&self) -> Result<NonZeroUsize> {
        NonZeroUsize::new(self.min_cluster_size).ok_or(ChutoroError::InvalidMinClusterSize {
            got: self.min_cluster_size,
        })
    }

    fn validate_for_batch(&self) -> Result<ValidatedBuilderConfig> {
        let min_cluster_size = self.validate_min_cluster_size()?;
        self.validate_batch_execution_strategy()?;
        Ok(ValidatedBuilderConfig { min_cluster_size })
    }

    #[cfg(feature = "cpu")]
    fn validate_for_session(&self) -> Result<ValidatedBuilderConfig> {
        let min_cluster_size = self.validate_min_cluster_size()?;
        self.validate_session_execution_strategy()?;
        Ok(ValidatedBuilderConfig { min_cluster_size })
    }

    fn validate_batch_execution_strategy(&self) -> Result<()> {
        if matches!(self.execution_strategy, ExecutionStrategy::GpuPreferred)
            && !cfg!(feature = "gpu")
        {
            return Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::GpuPreferred,
            });
        }

        Ok(())
    }

    #[cfg(feature = "cpu")]
    fn validate_session_execution_strategy(&self) -> Result<()> {
        if matches!(self.execution_strategy, ExecutionStrategy::GpuPreferred) {
            return Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::GpuPreferred,
            });
        }

        Ok(())
    }
}
