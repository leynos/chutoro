//! Chutoro core library.

use std::{collections::BTreeSet, num::NonZeroUsize};

use thiserror::Error;

/// An error produced by [`DataSource`] operations.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum DataSourceError {
    /// Requested index was outside the source's bounds.
    #[error("index {index} is out of bounds")]
    OutOfBounds { index: usize },
    /// Provided output buffer length did not match number of pairs.
    #[error("output buffer has length {out} but {expected} pairs were given")]
    OutputLengthMismatch { out: usize, expected: usize },
    /// Compared vectors had different dimensions.
    #[error("dimension mismatch: left={left}, right={right}")]
    DimensionMismatch { left: usize, right: usize },
}

/// Abstraction over a collection of items that can yield pairwise distances.
///
/// # Examples
/// ```
/// use chutoro_core::{DataSource, DataSourceError};
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
/// let src = Dummy(vec![1.0, 2.0, 4.0]);
/// assert_eq!(src.len(), 3);
/// assert_eq!(src.name(), "dummy");
/// assert_eq!(src.distance(0, 2)?, 3.0);
///
/// let pairs = vec![(0, 1), (1, 2)];
/// let mut out = vec![0.0; 2];
/// src.distance_batch(&pairs, &mut out)?;
/// assert_eq!(out, [1.0, 2.0]);
/// # Ok::<(), DataSourceError>(())
/// ```
pub trait DataSource {
    /// Returns number of items in the source.
    fn len(&self) -> usize;

    /// Returns whether the source contains no items.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{DataSource, DataSourceError};
    /// struct Empty;
    /// impl DataSource for Empty {
    ///     fn len(&self) -> usize { 0 }
    ///     fn name(&self) -> &str { "empty" }
    ///     fn distance(&self, _: usize, _: usize) -> Result<f32, DataSourceError> { Ok(0.0) }
    /// }
    /// let src = Empty;
    /// assert!(src.is_empty());
    /// ```
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a human-readable name.
    fn name(&self) -> &str;

    /// Computes the distance between two items.
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError>;

    /// Computes several distances at once, storing results in `out`.
    ///
    /// The default implementation calls [`distance`] for each pair.
    ///
    /// # Errors
    /// Returns `DataSourceError::OutputLengthMismatch` if `pairs.len() != out.len()`.
    ///
    /// If any pair fails, `out` is left unmodified.
    fn distance_batch(
        &self,
        pairs: &[(usize, usize)],
        out: &mut [f32],
    ) -> Result<(), DataSourceError> {
        if pairs.len() != out.len() {
            return Err(DataSourceError::OutputLengthMismatch {
                out: out.len(),
                expected: pairs.len(),
            });
        }
        // Compute into a temp buffer to keep `out` unchanged on error.
        let mut tmp = vec![0.0_f32; pairs.len()];
        for (idx, (i, j)) in pairs.iter().enumerate() {
            tmp[idx] = self.distance(*i, *j)?;
        }
        out.copy_from_slice(&tmp);
        Ok(())
    }
}

/// Indicates how [`Chutoro`] selects a compute backend when [`Chutoro::run`] is
/// invoked.
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

/// Error type produced when constructing or running [`Chutoro`].
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ChutoroError {
    /// Minimum cluster size must be greater than zero.
    #[error("min_cluster_size must be at least 1 (got {got})")]
    InvalidMinClusterSize { got: usize },
    /// The supplied [`DataSource`] contained no items.
    #[error("data source `{data_source}` contains no items")]
    EmptySource { data_source: String },
    /// The [`DataSource`] did not contain enough items for the configured
    /// `min_cluster_size`.
    #[error(
        "data source `{data_source}` has {items} items but min_cluster_size requires {min_cluster_size}"
    )]
    InsufficientItems {
        data_source: String,
        items: usize,
        min_cluster_size: NonZeroUsize,
    },
    /// The requested execution strategy is unavailable in the current build.
    #[error("the requested execution strategy {requested:?} is not available in this build")]
    BackendUnavailable { requested: ExecutionStrategy },
    /// A [`DataSource`] operation failed while running the algorithm.
    #[error("data source `{data_source}` failed: {error}")]
    DataSource {
        data_source: String,
        #[source]
        error: DataSourceError,
    },
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
}

#[derive(Debug, Clone)]
struct ChutoroConfig {
    min_cluster_size: NonZeroUsize,
    execution_strategy: ExecutionStrategy,
}

impl Default for ChutoroBuilder {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            execution_strategy: ExecutionStrategy::Auto,
        }
    }
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

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
    #[must_use]
    pub fn min_cluster_size(&self) -> usize {
        self.min_cluster_size
    }

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
    #[must_use]
    pub fn execution_strategy(&self) -> ExecutionStrategy {
        self.execution_strategy
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
    pub fn build(self) -> Result<Chutoro, ChutoroError> {
        let min_cluster_size = NonZeroUsize::new(self.min_cluster_size).ok_or(
            ChutoroError::InvalidMinClusterSize {
                got: self.min_cluster_size,
            },
        )?;
        Ok(Chutoro {
            config: ChutoroConfig {
                min_cluster_size,
                execution_strategy: self.execution_strategy,
            },
        })
    }
}

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
/// let chutoro = ChutoroBuilder::new().build()?;
/// let result = chutoro.run(&Dummy(vec![1.0, 2.0, 4.0]))?;
/// assert_eq!(result.assignments().len(), 3);
/// assert_eq!(result.cluster_count(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Chutoro {
    config: ChutoroConfig,
}

impl Chutoro {
    /// Returns the minimum cluster size configured for this instance.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ChutoroBuilder;
    ///
    /// let chutoro = ChutoroBuilder::new().with_min_cluster_size(9).build().unwrap();
    /// assert_eq!(chutoro.min_cluster_size().get(), 9);
    /// ```
    #[must_use]
    pub fn min_cluster_size(&self) -> NonZeroUsize {
        self.config.min_cluster_size
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
    ///     .unwrap();
    /// assert_eq!(chutoro.execution_strategy(), ExecutionStrategy::CpuOnly);
    /// ```
    #[must_use]
    pub fn execution_strategy(&self) -> ExecutionStrategy {
        self.config.execution_strategy
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
    /// let chutoro = ChutoroBuilder::new().build().unwrap();
    /// let result = chutoro.run(&Dummy(vec![1.0, 2.0, 4.0])).unwrap();
    /// assert_eq!(result.assignments().len(), 3);
    /// assert_eq!(result.cluster_count(), 1);
    /// ```
    pub fn run<D: DataSource>(&self, source: &D) -> Result<ClusteringResult, ChutoroError> {
        let backend = self.resolve_backend()?;
        let len = source.len();
        if len == 0 {
            return Err(ChutoroError::EmptySource {
                data_source: source.name().to_owned(),
            });
        }
        if len < self.config.min_cluster_size.get() {
            return Err(ChutoroError::InsufficientItems {
                data_source: source.name().to_owned(),
                items: len,
                min_cluster_size: self.config.min_cluster_size,
            });
        }

        match backend {
            ExecutionBackend::Cpu => self.run_cpu(source, len),
        }
    }

    fn resolve_backend(&self) -> Result<ExecutionBackend, ChutoroError> {
        match self.config.execution_strategy {
            ExecutionStrategy::Auto | ExecutionStrategy::CpuOnly => Ok(ExecutionBackend::Cpu),
            ExecutionStrategy::GpuPreferred => Err(ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::GpuPreferred,
            }),
        }
    }

    fn run_cpu<D: DataSource>(
        &self,
        _source: &D,
        items: usize,
    ) -> Result<ClusteringResult, ChutoroError> {
        let assignments = vec![ClusterId::new(0); items];
        Ok(ClusteringResult::from_assignments(assignments))
    }
}

#[derive(Debug, Clone, Copy)]
enum ExecutionBackend {
    Cpu,
}

/// Represents the output of a [`Chutoro::run`] invocation.
///
/// # Examples
/// ```
/// use chutoro_core::{ClusteringResult, ClusterId};
///
/// let result = ClusteringResult::from_assignments(vec![ClusterId::new(0), ClusterId::new(1)]);
/// assert_eq!(result.assignments().len(), 2);
/// assert_eq!(result.cluster_count(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusteringResult {
    assignments: Vec<ClusterId>,
}

impl ClusteringResult {
    /// Builds a result from explicit cluster assignments.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ClusteringResult, ClusterId};
    ///
    /// let result = ClusteringResult::from_assignments(vec![ClusterId::new(0)]);
    /// assert_eq!(result.cluster_count(), 1);
    /// ```
    #[must_use]
    pub fn from_assignments(assignments: Vec<ClusterId>) -> Self {
        Self { assignments }
    }

    /// Returns the assignments in insertion order.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ClusteringResult, ClusterId};
    ///
    /// let result = ClusteringResult::from_assignments(vec![ClusterId::new(0)]);
    /// assert_eq!(result.assignments()[0].get(), 0);
    /// ```
    #[must_use]
    pub fn assignments(&self) -> &[ClusterId] {
        &self.assignments
    }

    /// Counts how many distinct clusters exist within the assignments.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ClusteringResult, ClusterId};
    ///
    /// let result = ClusteringResult::from_assignments(vec![ClusterId::new(0), ClusterId::new(0)]);
    /// assert_eq!(result.cluster_count(), 1);
    /// ```
    #[must_use]
    pub fn cluster_count(&self) -> usize {
        self.assignments
            .iter()
            .map(|id| id.get())
            .collect::<BTreeSet<_>>()
            .len()
    }
}

/// Identifier assigned to a cluster.
///
/// # Examples
/// ```
/// use chutoro_core::ClusterId;
///
/// let id = ClusterId::new(4);
/// assert_eq!(id.get(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClusterId(u64);

impl ClusterId {
    /// Creates a new cluster identifier.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ClusterId;
    ///
    /// let id = ClusterId::new(2);
    /// assert_eq!(id.get(), 2);
    /// ```
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying numeric identifier.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ClusterId;
    ///
    /// let id = ClusterId::new(7);
    /// assert_eq!(id.get(), 7);
    /// ```
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}
