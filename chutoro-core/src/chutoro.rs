//! Core clustering orchestration for the Chutoro library.
//!
//! Provides the [`Chutoro`] runtime entry point and helpers for selecting
//! execution backends and wrapping data-source failures.

use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    Result, builder::ExecutionStrategy, datasource::DataSource, error::ChutoroError,
    result::ClusteringResult,
};
use tracing::{instrument, warn};

const CPU_PATH_AVAILABLE: bool = cfg!(feature = "cpu");
// The `gpu` feature currently exposes the orchestration surface only;
// no accelerated implementation ships yet.
const GPU_PATH_AVAILABLE: bool = false;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendChoice {
    Cpu,
    Gpu,
}

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
    pub fn min_cluster_size(&self) -> NonZeroUsize {
        self.min_cluster_size
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
        err,
        skip(self, source),
        fields(
            data_source = %source.name(),
            items = items,
            min_cluster_size = %self.min_cluster_size,
            strategy = ?self.execution_strategy
        ),
    )]
    fn run_with_len<D: DataSource + Sync>(
        &self,
        source: &D,
        items: usize,
    ) -> Result<ClusteringResult> {
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

        match self.choose_backend() {
            BackendChoice::Cpu => self.run_cpu(source, items),
            BackendChoice::Gpu => self.run_gpu(source, items),
        }
    }

    fn choose_backend(&self) -> BackendChoice {
        match self.execution_strategy {
            ExecutionStrategy::Auto => {
                if CPU_PATH_AVAILABLE {
                    BackendChoice::Cpu
                } else {
                    BackendChoice::Gpu
                }
            }
            ExecutionStrategy::CpuOnly => BackendChoice::Cpu,
            ExecutionStrategy::GpuPreferred => BackendChoice::Gpu,
        }
    }

    /// Execute the CPU FISHDBC pipeline; available with the `cpu` feature.
    #[instrument(
        name = "core.run_cpu",
        err,
        skip(self, source),
        fields(items = items, min_cluster_size = %self.min_cluster_size),
    )]
    fn run_cpu<D: DataSource + Sync>(&self, source: &D, items: usize) -> Result<ClusteringResult> {
        #[cfg(feature = "cpu")]
        {
            crate::cpu_pipeline::run_cpu_pipeline_with_len(source, items, self.min_cluster_size)
        }
        #[cfg(not(feature = "cpu"))]
        {
            let _ = (source, items);
            Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::CpuOnly,
            })
        }
    }

    fn run_gpu<D: DataSource + Sync>(
        &self,
        _source: &D,
        _items: usize,
    ) -> Result<ClusteringResult> {
        Err(ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred,
        })
    }

    fn backend_unavailable_error(&self) -> Option<ChutoroError> {
        let unavailable = self.is_backend_unavailable();

        unavailable.then_some(ChutoroError::BackendUnavailable {
            requested: self.execution_strategy,
        })
    }

    fn is_backend_unavailable(&self) -> bool {
        match self.execution_strategy {
            ExecutionStrategy::Auto => !(CPU_PATH_AVAILABLE || GPU_PATH_AVAILABLE),
            ExecutionStrategy::CpuOnly => !CPU_PATH_AVAILABLE,
            ExecutionStrategy::GpuPreferred => !GPU_PATH_AVAILABLE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_preferred_requires_gpu_feature() {
        let chutoro = Chutoro::new(
            NonZeroUsize::new(1).expect("literal 1 is non-zero"),
            ExecutionStrategy::GpuPreferred,
        );
        let err = chutoro.backend_unavailable_error();
        assert!(matches!(
            err,
            Some(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::GpuPreferred
            })
        ));
    }

    #[test]
    fn backend_available_when_features_enabled() {
        if cfg!(feature = "cpu") {
            for strategy in [ExecutionStrategy::Auto, ExecutionStrategy::CpuOnly] {
                let chutoro = Chutoro::new(
                    NonZeroUsize::new(1).expect("literal 1 is non-zero"),
                    strategy,
                );
                assert!(chutoro.backend_unavailable_error().is_none());
            }
        }

        let chutoro = Chutoro::new(
            NonZeroUsize::new(1).expect("literal 1 is non-zero"),
            ExecutionStrategy::GpuPreferred,
        );
        assert!(matches!(
            chutoro.backend_unavailable_error(),
            Some(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::GpuPreferred
            })
        ));
    }
}
