//! Core clustering orchestration for the Chutoro library.
//!
//! Provides the [`Chutoro`] runtime entry point and helpers for selecting
//! execution backends and wrapping data-source failures.

use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    Result,
    backend::{BackendChoice, backend_label, choose_backend, is_backend_unavailable},
    builder::ExecutionStrategy,
    datasource::DataSource,
    error::ChutoroError,
    execution_config::ExecutionConfig,
    result::ClusteringResult,
};
#[cfg(feature = "cpu")]
use tracing::debug;
use tracing::{instrument, warn};

/// Entry point for running the clustering pipeline.
///
/// # Examples
/// ```rust,no_run
/// use chutoro_core::{ChutoroBuilder, DataSource, DataSourceError};
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
/// let chutoro = ChutoroBuilder::new()
///     .with_min_cluster_size(3)
///     .build()
///     .expect("builder must succeed");
/// let result = chutoro
///     .run(&Dummy(vec![1.0, 2.0, 4.0]))
///     .expect("run must succeed");
/// assert_eq!(result.assignments().len(), 3);
/// assert_eq!(result.cluster_count(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Chutoro {
    /// Validated clustering and CPU HNSW policy selected by the builder.
    execution_config: ExecutionConfig,
    /// Backend-selection policy chosen by the builder.
    execution_strategy: ExecutionStrategy,
    /// Optional guard for estimated peak memory consumption.
    max_bytes: Option<u64>,
}

impl Chutoro {
    /// Construct an orchestrator from already validated builder state.
    pub(crate) const fn new(
        execution_config: ExecutionConfig,
        execution_strategy: ExecutionStrategy,
        max_bytes: Option<u64>,
    ) -> Self {
        Self {
            execution_config,
            execution_strategy,
            max_bytes,
        }
    }

    /// Returns the minimum cluster size configured for this instance.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// let chutoro = ChutoroBuilder::new()
    ///     .with_min_cluster_size(9)
    ///     .build()
    ///     .expect("builder must accept non-zero min_cluster_size");
    /// assert_eq!(chutoro.min_cluster_size().get(), 9);
    /// ```
    #[must_use]
    pub const fn min_cluster_size(&self) -> NonZeroUsize {
        self.execution_config.min_cluster_size()
    }

    /// Returns the HNSW parameters configured for CPU execution.
    #[cfg(feature = "cpu")]
    pub(crate) const fn hnsw_params(&self) -> &crate::HnswParams {
        self.execution_config.hnsw_params()
    }

    /// Returns the execution strategy that will be used when running.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use chutoro_core::{ChutoroBuilder, ExecutionStrategy};
    ///
    /// let chutoro = ChutoroBuilder::new()
    ///     .with_execution_strategy(ExecutionStrategy::CpuOnly)
    ///     .build()
    ///     .expect("builder must apply execution strategy");
    /// assert_eq!(chutoro.execution_strategy(), ExecutionStrategy::CpuOnly);
    /// ```
    #[must_use]
    pub const fn execution_strategy(&self) -> ExecutionStrategy {
        self.execution_strategy
    }

    /// Returns the optional memory limit in bytes, if configured.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// let chutoro = ChutoroBuilder::new()
    ///     .with_max_bytes(1_073_741_824)
    ///     .build()
    ///     .expect("builder must succeed");
    /// assert_eq!(chutoro.max_bytes(), Some(1_073_741_824));
    /// ```
    #[rustfmt::skip]
    #[must_use]
    pub const fn max_bytes(&self) -> Option<u64> { self.max_bytes }

    /// Executes the clustering pipeline against the provided [`DataSource`].
    ///
    /// # Errors
    /// Returns [`ChutoroError::EmptySource`] when the [`DataSource`] is empty,
    /// [`ChutoroError::InsufficientItems`] when it does not satisfy
    /// `min_cluster_size`, [`ChutoroError::MemoryLimitExceeded`] when the
    /// estimated memory exceeds `max_bytes`, and
    /// [`ChutoroError::BackendUnavailable`] when the requested backend is not
    /// compiled in the current build.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use chutoro_core::{ChutoroBuilder, DataSource, DataSourceError};
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
    /// let chutoro = ChutoroBuilder::new()
    ///     .with_min_cluster_size(3)
    ///     .build()
    ///     .expect("builder must succeed");
    /// let result = chutoro
    ///     .run(&Dummy(vec![1.0, 2.0, 4.0]))
    ///     .expect("run must succeed");
    /// assert_eq!(result.assignments().len(), 3);
    /// assert_eq!(result.cluster_count(), 1);
    /// ```
    pub fn run<D: DataSource + Sync>(&self, source: &D) -> Result<ClusteringResult> {
        let items = source.len();
        self.run_with_len(source, items)
    }

    #[instrument(
        name = "core.run",
        skip(self, source),
        fields(
            items = items,
            min_cluster_size = %self.min_cluster_size(),
            strategy = ?self.execution_strategy,
            backend = backend_label(self.execution_strategy)
        ),
    )]
    /// Run clustering after the caller has measured the source length.
    fn run_with_len<D: DataSource + Sync>(
        &self,
        source: &D,
        items: usize,
    ) -> Result<ClusteringResult> {
        let backend = backend_label(self.execution_strategy);
        if items == 0 {
            let error = ChutoroError::EmptySource {
                data_source: Arc::from(source.name()),
            };
            warn!(
                backend,
                error_code = error.code().as_str(),
                "data source is empty, returning error"
            );
            return Self::record_batch_result(backend, Err(error));
        }
        if items < self.min_cluster_size().get() {
            let error = ChutoroError::InsufficientItems {
                data_source: Arc::from(source.name()),
                items,
                min_cluster_size: self.min_cluster_size(),
            };
            warn!(
                backend,
                error_code = error.code().as_str(),
                "data source has insufficient items for configured cluster size"
            );
            return Self::record_batch_result(backend, Err(error));
        }
        if let Some(err) = self.backend_unavailable_error() {
            warn!(
                backend,
                error_code = err.code().as_str(),
                "requested batch backend is unavailable"
            );
            return Self::record_batch_result(backend, Err(err));
        }

        self.record_batch_resources(items);

        if let Err(error) = self.check_memory_limit(source, items) {
            return Self::record_batch_result(backend, Err(error));
        }

        let result = match choose_backend(self.execution_strategy) {
            BackendChoice::Cpu => self.run_cpu(source, items),
            BackendChoice::Gpu => Self::run_gpu(source, items),
        };
        if let Err(error) = &result {
            warn!(
                backend,
                error_code = error.code().as_str(),
                "batch execution failed after precondition checks"
            );
        }
        Self::record_batch_result(backend, result)
    }

    /// Records the stable outcome dimensions for a completed batch attempt.
    #[cfg(feature = "metrics")]
    fn record_batch_result<T>(backend: &'static str, result: Result<T>) -> Result<T> {
        crate::batch_metrics::record_outcome(backend, &result);
        result
    }

    /// Returns a batch result unchanged when metrics are unavailable.
    #[cfg(not(feature = "metrics"))]
    const fn record_batch_result<T>(_backend: &'static str, result: Result<T>) -> Result<T> {
        result
    }

    /// Records bounded CPU resource observations before the memory guard runs.
    #[cfg(all(feature = "cpu", feature = "metrics"))]
    fn record_batch_resources(&self, items: usize) {
        let hnsw_params = self.hnsw_params();
        crate::batch_metrics::record_cpu_resources(
            hnsw_params.max_connections(),
            hnsw_params.effective_ef_construction(items),
            crate::memory::estimate_peak_bytes_for_hnsw_params(items, hnsw_params),
            self.max_bytes,
        );
    }

    /// Does not record CPU resources when the required features are unavailable.
    #[cfg(not(all(feature = "cpu", feature = "metrics")))]
    const fn record_batch_resources(&self, items: usize) {
        let _ = (self, items);
    }

    /// Returns an error if the estimated peak memory exceeds `max_bytes`.
    fn check_memory_limit<D: DataSource>(&self, source: &D, items: usize) -> Result<()> {
        let Some(limit) = self.max_bytes else {
            return Ok(());
        };

        #[cfg(feature = "cpu")]
        let estimated =
            crate::memory::estimate_peak_bytes_for_hnsw_params(items, self.hnsw_params());
        #[cfg(not(feature = "cpu"))]
        let estimated = 0;

        #[cfg(feature = "cpu")]
        debug!(
            backend = "cpu",
            max_connections = self.hnsw_params().max_connections(),
            configured_ef_construction = self.hnsw_params().ef_construction(),
            effective_ef_construction = self.hnsw_params().effective_ef_construction(items),
            estimated_bytes = estimated,
            max_bytes = limit,
            "checked CPU memory limit"
        );

        if estimated > limit {
            let error = ChutoroError::MemoryLimitExceeded {
                data_source: Arc::from(source.name()),
                point_count: items,
                estimated_bytes: estimated,
                max_bytes: limit,
                estimated_display: Arc::from(crate::memory::format_bytes(estimated)),
                limit_display: Arc::from(crate::memory::format_bytes(limit)),
            };
            #[cfg(feature = "cpu")]
            warn!(
                backend = "cpu",
                max_connections = self.hnsw_params().max_connections(),
                estimated_bytes = estimated,
                max_bytes = limit,
                error_code = error.code().as_str(),
                "CPU memory estimate exceeds configured limit"
            );
            return Err(error);
        }
        Ok(())
    }

    /// Execute the CPU FISHDBC pipeline; available with the `cpu` feature.
    #[instrument(
        name = "core.run_cpu",
        skip(self, source),
        fields(items = items, min_cluster_size = %self.min_cluster_size()),
    )]
    fn run_cpu<D: DataSource + Sync>(&self, source: &D, items: usize) -> Result<ClusteringResult> {
        #[cfg(feature = "cpu")]
        {
            crate::cpu_pipeline::run_cpu_pipeline_with_len(
                source,
                items,
                self.min_cluster_size(),
                self.hnsw_params(),
            )
        }
        #[cfg(not(feature = "cpu"))]
        {
            let _ = (source, items);
            Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::CpuOnly,
            })
        }
    }

    /// Return the current GPU-backend-unavailable result.
    const fn run_gpu<D: DataSource + Sync>(_source: &D, _items: usize) -> Result<ClusteringResult> {
        Err(ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred,
        })
    }

    /// Return the unavailable-backend error when the strategy cannot execute.
    fn backend_unavailable_error(&self) -> Option<ChutoroError> {
        let unavailable = is_backend_unavailable(self.execution_strategy);

        unavailable.then_some(ChutoroError::BackendUnavailable {
            requested: self.execution_strategy,
        })
    }
}

#[cfg(test)]
#[path = "chutoro_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cpu"))]
#[path = "chutoro/properties.rs"]
mod properties;
