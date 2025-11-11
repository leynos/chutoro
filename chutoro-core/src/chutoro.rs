//! Core clustering orchestration for the Chutoro library.
//!
//! Provides the [`Chutoro`] runtime entry point and helpers for selecting
//! execution backends and wrapping data-source failures.

use std::{num::NonZeroUsize, sync::Arc};

#[cfg(feature = "skeleton")]
use crate::error::DataSourceError;
#[cfg(feature = "skeleton")]
use crate::result::ClusterId;
use crate::{
    Result, builder::ExecutionStrategy, datasource::DataSource, error::ChutoroError,
    result::ClusteringResult,
};
#[cfg(feature = "skeleton")]
use tracing::info;
use tracing::{instrument, warn};

#[cfg(feature = "skeleton")]
type DataSourceResult<T> = core::result::Result<T, DataSourceError>;

/// Entry point for running the clustering pipeline.
///
/// # Examples
/// ```
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
    min_cluster_size: NonZeroUsize,
    execution_strategy: ExecutionStrategy,
}

impl Chutoro {
    pub(crate) fn new(
        min_cluster_size: NonZeroUsize,
        execution_strategy: ExecutionStrategy,
    ) -> Self {
        Self {
            min_cluster_size,
            execution_strategy,
        }
    }

    /// Returns the minimum cluster size configured for this instance.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// let chutoro = ChutoroBuilder::new()
    ///     .with_min_cluster_size(9)
    ///     .build()
    ///     .expect("builder must accept non-zero min_cluster_size");
    /// assert_eq!(chutoro.min_cluster_size().get(), 9);
    /// ```
    #[must_use]
    pub fn min_cluster_size(&self) -> NonZeroUsize {
        self.min_cluster_size
    }

    /// Returns the execution strategy that will be used when running.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ChutoroBuilder, ExecutionStrategy};
    ///
    /// let chutoro = ChutoroBuilder::new()
    ///     .with_execution_strategy(ExecutionStrategy::CpuOnly)
    ///     .build()
    ///     .expect("builder must apply execution strategy");
    /// assert_eq!(chutoro.execution_strategy(), ExecutionStrategy::CpuOnly);
    /// ```
    #[must_use]
    pub fn execution_strategy(&self) -> ExecutionStrategy {
        self.execution_strategy
    }

    /// Executes the clustering pipeline against the provided [`DataSource`].
    ///
    /// # Errors
    /// Returns [`ChutoroError::EmptySource`] when the [`DataSource`] is empty,
    /// [`ChutoroError::InsufficientItems`] when it does not satisfy
    /// `min_cluster_size`, and [`ChutoroError::BackendUnavailable`] when the
    /// requested backend is not compiled in the current build.
    ///
    /// # Examples
    /// ```
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
    pub fn run<D: DataSource>(&self, source: &D) -> Result<ClusteringResult> {
        let items = source.len();
        self.run_with_len(source, items)
    }

    #[instrument(
        name = "core.run",
        err,
        skip(self, source),
        fields(
            data_source = %source.name(),
            items = items,
            min_cluster_size = %self.min_cluster_size,
            strategy = ?self.execution_strategy
        ),
    )]
    fn run_with_len<D: DataSource>(&self, source: &D, items: usize) -> Result<ClusteringResult> {
        if items == 0 {
            warn!(
                data_source = source.name(),
                "data source is empty, returning error"
            );
            return Err(ChutoroError::EmptySource {
                data_source: Arc::from(source.name()),
            });
        }
        if items < self.min_cluster_size.get() {
            return Err(ChutoroError::InsufficientItems {
                data_source: Arc::from(source.name()),
                items,
                min_cluster_size: self.min_cluster_size,
            });
        }
        if let Some(err) = self.backend_unavailable_error() {
            return Err(err);
        }

        match self.execution_strategy {
            // GPU + skeleton: route Auto and GpuPreferred to GPU path
            #[cfg(all(feature = "gpu", feature = "skeleton"))]
            ExecutionStrategy::Auto | ExecutionStrategy::GpuPreferred => {
                self.run_gpu(source, items)
            }

            // GPU only (no skeleton): route both strategies to the GPU stub
            #[cfg(all(feature = "gpu", not(feature = "skeleton")))]
            ExecutionStrategy::GpuPreferred => self.run_gpu(source, items),
            #[cfg(all(feature = "gpu", not(feature = "skeleton")))]
            ExecutionStrategy::Auto => self.run_gpu(source, items),
            #[cfg(all(feature = "skeleton", not(feature = "gpu")))]
            ExecutionStrategy::Auto => {
                self.wrap_datasource_error(source, self.run_cpu(source, items))
            }
            #[cfg(all(not(feature = "skeleton"), not(feature = "gpu")))]
            ExecutionStrategy::Auto => Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::Auto,
            }),
            #[cfg(feature = "skeleton")]
            ExecutionStrategy::CpuOnly => {
                self.wrap_datasource_error(source, self.run_cpu(source, items))
            }
            #[cfg(not(feature = "skeleton"))]
            ExecutionStrategy::CpuOnly => Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::CpuOnly,
            }),
            #[cfg(not(feature = "gpu"))]
            ExecutionStrategy::GpuPreferred => Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::GpuPreferred,
            }),
        }
    }

    /// Execute the CPU walking skeleton; available with the `skeleton` feature.
    #[cfg(feature = "skeleton")]
    #[instrument(
        name = "core.run_cpu",
        err,
        skip(self, _source),
        fields(items = items, min_cluster_size = %self.min_cluster_size),
    )]
    fn run_cpu<D: DataSource>(
        &self,
        _source: &D,
        items: usize,
    ) -> DataSourceResult<ClusteringResult> {
        // FIXME(#12): This is a walking skeleton implementation that partitions items into
        // fixed-size buckets based on min_cluster_size. Replace with HNSW + MST +
        // hierarchy extraction as per the FISHDBC algorithm design.
        let cluster_span = self.min_cluster_size.get();
        let assignments = (0..items)
            .map(|idx| ClusterId::new((idx / cluster_span) as u64))
            .collect();
        let result = ClusteringResult::from_assignments(assignments);
        info!(clusters = result.cluster_count(), "cpu execution completed");
        Ok(result)
    }

    /// Execute the GPU walking skeleton; requires the `gpu` and `skeleton` features.
    #[cfg(all(feature = "gpu", feature = "skeleton"))]
    fn run_gpu<D: DataSource>(&self, source: &D, items: usize) -> Result<ClusteringResult> {
        // TODO(#13): Replace with the real GPU backend once implemented. Until then we
        // reuse the CPU walking skeleton to exercise the orchestration path.
        self.wrap_datasource_error(source, self.run_cpu(source, items))
    }

    /// Fail when GPU execution is requested without the `skeleton` feature.
    #[cfg(all(feature = "gpu", not(feature = "skeleton")))]
    fn run_gpu<D: DataSource>(&self, _source: &D, _items: usize) -> Result<ClusteringResult> {
        // We intentionally fail fast when the walking skeleton is disabled so GPU
        // builds do not accidentally ship the placeholder CPU path.
        Err(ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred,
        })
    }

    fn backend_unavailable_error(&self) -> Option<ChutoroError> {
        const CPU_PATH_AVAILABLE: bool = cfg!(feature = "skeleton");
        const GPU_PATH_AVAILABLE: bool = cfg!(all(feature = "gpu", feature = "skeleton"));

        let unavailable = match self.execution_strategy {
            ExecutionStrategy::CpuOnly | ExecutionStrategy::Auto => !CPU_PATH_AVAILABLE,
            ExecutionStrategy::GpuPreferred => !GPU_PATH_AVAILABLE,
        };

        unavailable.then_some(ChutoroError::BackendUnavailable {
            requested: self.execution_strategy,
        })
    }

    #[cfg(feature = "skeleton")]
    fn wrap_datasource_error<D: DataSource>(
        &self,
        source: &D,
        result: DataSourceResult<ClusteringResult>,
    ) -> Result<ClusteringResult> {
        result.map_err(|error| ChutoroError::DataSource {
            data_source: Arc::from(source.name()),
            error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "skeleton"))]
    #[test]
    fn backend_error_without_skeleton() {
        let chutoro = Chutoro::new(
            NonZeroUsize::new(1).expect("literal 1 is non-zero"),
            ExecutionStrategy::Auto,
        );
        assert!(chutoro.backend_unavailable_error().is_some());

        let cpu_only = Chutoro::new(
            NonZeroUsize::new(1).expect("literal 1 is non-zero"),
            ExecutionStrategy::CpuOnly,
        );
        assert!(cpu_only.backend_unavailable_error().is_some());
    }

    #[cfg(all(feature = "skeleton", not(feature = "gpu")))]
    #[test]
    fn gpu_preferred_requires_gpu_feature() {
        let chutoro = Chutoro::new(
            NonZeroUsize::new(1).expect("literal 1 is non-zero"),
            ExecutionStrategy::GpuPreferred,
        );
        let err = chutoro.backend_unavailable_error();
        assert!(matches!(err, Some(ChutoroError::BackendUnavailable { .. })));

        let auto = Chutoro::new(
            NonZeroUsize::new(1).expect("literal 1 is non-zero"),
            ExecutionStrategy::Auto,
        );
        assert!(auto.backend_unavailable_error().is_none());
    }

    #[cfg(all(feature = "gpu", feature = "skeleton"))]
    #[test]
    fn backend_available_when_features_enabled() {
        for strategy in [
            ExecutionStrategy::Auto,
            ExecutionStrategy::CpuOnly,
            ExecutionStrategy::GpuPreferred,
        ] {
            let chutoro = Chutoro::new(
                NonZeroUsize::new(1).expect("literal 1 is non-zero"),
                strategy,
            );
            assert!(chutoro.backend_unavailable_error().is_none());
        }
    }
}
