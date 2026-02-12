//! Shared test utilities for `chutoro-core`.

use chutoro_test_support::ci::property_test_profile::ProptestRunProfile;
use proptest::test_runner::Config as ProptestConfig;

use crate::{datasource::DataSource, error::DataSourceError};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Builds a standard proptest configuration from the shared CI profile.
///
/// This keeps property suites aligned on the same `PROGTEST_CASES` and
/// `CHUTORO_PBT_FORK` interpretation.
#[must_use]
pub(crate) fn suite_proptest_config(default_cases: u32) -> ProptestConfig {
    let profile = ProptestRunProfile::load(default_cases, false);
    ProptestConfig {
        cases: profile.cases(),
        fork: profile.fork(),
        ..ProptestConfig::default()
    }
}

/// [`DataSource`] implementation that records distance invocations for tests.
///
/// # Examples
/// ```ignore
/// use std::sync::{Arc, atomic::AtomicUsize};
/// use chutoro_core::DataSource;
/// use chutoro_core::test_utils::CountingSource;
///
/// let counter = Arc::new(AtomicUsize::new(0));
/// let source = CountingSource::new(vec![0.0, 1.0], Arc::clone(&counter));
/// assert_eq!(source.distance(0, 1)?, 1.0);
/// assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 1);
/// # Ok::<(), chutoro_core::error::DataSourceError>(())
/// ```
#[derive(Clone)]
pub(crate) struct CountingSource {
    data: Vec<f32>,
    calls: Arc<AtomicUsize>,
    name: &'static str,
}

impl CountingSource {
    /// Creates a counting source with the default "counting" name.
    #[must_use]
    pub(crate) fn new(data: Vec<f32>, calls: Arc<AtomicUsize>) -> Self {
        Self::with_name("counting", data, calls)
    }

    /// Creates a counting source with a specific display name.
    #[must_use]
    pub(crate) fn with_name(name: &'static str, data: Vec<f32>, calls: Arc<AtomicUsize>) -> Self {
        Self { data, calls, name }
    }

    /// Returns the backing distance counter for assertions.
    #[must_use]
    pub(crate) fn calls(&self) -> &Arc<AtomicUsize> {
        &self.calls
    }

    /// Returns an immutable view over the stored values.
    #[must_use]
    pub(crate) fn data(&self) -> &[f32] {
        &self.data
    }
}

impl DataSource for CountingSource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        self.name
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let a = self
            .data
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let b = self
            .data
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok((a - b).abs())
    }
}
