use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    builder::ExecutionStrategy,
    datasource::DataSource,
    error::{ChutoroError, DataSourceError},
    result::{ClusterId, ClusteringResult},
};

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
    pub fn run<D: DataSource>(&self, source: &D) -> Result<ClusteringResult, ChutoroError> {
        let len = source.len();
        if len == 0 {
            return Err(ChutoroError::EmptySource {
                data_source: Arc::from(source.name()),
            });
        }
        if len < self.min_cluster_size.get() {
            return Err(ChutoroError::InsufficientItems {
                data_source: Arc::from(source.name()),
                items: len,
                min_cluster_size: self.min_cluster_size,
            });
        }

        match self.execution_strategy {
            #[cfg(feature = "skeleton")]
            ExecutionStrategy::Auto | ExecutionStrategy::CpuOnly => {
                self.wrap_datasource_error(source, self.run_cpu(source, len))
            }
            #[cfg(not(feature = "skeleton"))]
            ExecutionStrategy::Auto | ExecutionStrategy::CpuOnly => {
                Err(ChutoroError::BackendUnavailable {
                    requested: self.execution_strategy,
                })
            }
            #[cfg(feature = "gpu")]
            ExecutionStrategy::GpuPreferred => self.run_gpu(source, len),
            #[cfg(not(feature = "gpu"))]
            ExecutionStrategy::GpuPreferred => Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::GpuPreferred,
            }),
        }
    }

    #[cfg(feature = "skeleton")]
    fn run_cpu<D: DataSource>(
        &self,
        _source: &D,
        items: usize,
    ) -> Result<ClusteringResult, DataSourceError> {
        // FIXME: This is a walking skeleton implementation that partitions items into
        // fixed-size buckets based on min_cluster_size. Replace with HNSW + MST +
        // hierarchy extraction as per the FISHDBC algorithm design.
        let cluster_span = self.min_cluster_size.get();
        let assignments = (0..items)
            .map(|idx| ClusterId::new((idx / cluster_span) as u64))
            .collect();
        Ok(ClusteringResult::from_assignments(assignments))
    }

    #[cfg(all(feature = "gpu", feature = "skeleton"))]
    fn run_gpu<D: DataSource>(
        &self,
        source: &D,
        items: usize,
    ) -> Result<ClusteringResult, ChutoroError> {
        // TODO: Replace with the real GPU backend once implemented. Until then we
        // reuse the CPU walking skeleton to exercise the orchestration path.
        self.wrap_datasource_error(source, self.run_cpu(source, items))
    }

    #[cfg(all(feature = "gpu", not(feature = "skeleton")))]
    fn run_gpu<D: DataSource>(
        &self,
        _source: &D,
        _items: usize,
    ) -> Result<ClusteringResult, ChutoroError> {
        // We intentionally fail fast when the walking skeleton is disabled so GPU
        // builds do not accidentally ship the placeholder CPU path.
        Err(ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred,
        })
    }

    fn wrap_datasource_error<D: DataSource>(
        &self,
        source: &D,
        result: Result<ClusteringResult, DataSourceError>,
    ) -> Result<ClusteringResult, ChutoroError> {
        result.map_err(|error| ChutoroError::DataSource {
            data_source: Arc::from(source.name()),
            error,
        })
    }
}
