//! Shared test utilities for `chutoro-core`.

use chutoro_test_support::ci::property_test_profile::{PROPTEST_RNG_SEED, ProptestRunProfile};
use proptest::test_runner::{Config as ProptestConfig, RngSeed};

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
        // Rejects are a budget for the whole run, so a deep run needs one in
        // proportion to the cases it asks for. See #260.
        max_global_rejects: profile.max_global_rejects(),
        rng_seed: RngSeed::Fixed(PROPTEST_RNG_SEED),
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
        Ok(a.mul_add(1.0, std::ops::Neg::neg(*b)).abs())
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the shared suite configuration.

    use super::*;
    use chutoro_test_support::ci::property_test_profile::{
        DEFAULT_MAX_GLOBAL_REJECTS, max_global_rejects_for,
    };

    /// The shared suite config sizes its reject budget from its case count.
    ///
    /// proptest's default is a flat 1024 for the whole run however many cases
    /// are asked for, and every `proptest!` suite in this crate takes its
    /// configuration from here. Dropping the field would put the weekly lane
    /// back on that default, which the HNSW suites exhausted the first time
    /// they ever drew a case (#260).
    ///
    /// The assertion is against the rule rather than a literal, so it holds
    /// whatever `PROPTEST_CASES` the surrounding run sets.
    #[test]
    fn the_shared_suite_config_scales_its_reject_budget() {
        let config = suite_proptest_config(25_000);

        assert!(
            max_global_rejects_for(config.cases) > DEFAULT_MAX_GLOBAL_REJECTS,
            "the fixture must ask for a run deep enough that the floor is not \
             the answer, or this test passes whether the budget is derived or \
             left on proptest's default"
        );

        assert_eq!(
            config.max_global_rejects,
            max_global_rejects_for(config.cases),
            "a deep run left on proptest's flat default aborts before it finishes"
        );
    }
}
