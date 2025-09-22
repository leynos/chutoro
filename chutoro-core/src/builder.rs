//! Builder utilities for configuring Chutoro orchestration.
//!
//! Exposes the execution strategy selection surface and builder validation used before constructing [`Chutoro`] instances.

use std::num::NonZeroUsize;

use crate::{Result, chutoro::Chutoro, error::ChutoroError};

/// Indicates how [`Chutoro`] selects a compute backend when [`Chutoro::run`] is
/// invoked.
///
/// `Auto` resolves backends deterministically. Today it maps to the CPU
/// walking skeleton because no GPU implementation ships with the crate.
/// Once a GPU backend lands it will select the GPU when that feature is
/// enabled and fall back to the CPU otherwise so behaviour stays stable
/// across builds.
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
    pub fn build(self) -> Result<Chutoro> {
        let min_cluster_size = NonZeroUsize::new(self.min_cluster_size).ok_or(
            ChutoroError::InvalidMinClusterSize {
                got: self.min_cluster_size,
            },
        )?;

        Ok(Chutoro::new(min_cluster_size, self.execution_strategy))
    }
}
