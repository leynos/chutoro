//! Shared test utilities for `chutoro-core`.

use chutoro_test_support::ci::property_test_profile::{PROPTEST_RNG_SEED, ProptestRunProfile};
use proptest::test_runner::{Config as ProptestConfig, RngSeed};

use crate::{
    datasource::{DataSource, MetricDescriptor},
    error::DataSourceError,
};
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

/// A small in-memory source for tests that need scalar absolute distances.
///
/// # Examples
/// ```ignore
/// use chutoro_core::test_utils::NumericTestSource;
///
/// let source = NumericTestSource::with_len("dummy", 3);
/// assert_eq!(source.name(), "dummy");
/// assert_eq!(source.distance(0, 2)?, 2.0);
/// # Ok::<(), chutoro_core::error::DataSourceError>(())
/// ```
#[derive(Clone, Debug)]
pub(crate) struct NumericTestSource {
    data: Vec<f32>,
    name: &'static str,
    metric_descriptor: Option<MetricDescriptor>,
}

impl NumericTestSource {
    /// Creates a numeric test source with the supplied values and name.
    ///
    /// # Examples
    /// ```ignore
    /// use chutoro_core::test_utils::NumericTestSource;
    ///
    /// let source = NumericTestSource::new("dummy", vec![1.0, 3.0]);
    /// assert_eq!(source.len(), 2);
    /// ```
    #[must_use]
    pub(crate) fn new(name: &'static str, data: Vec<f32>) -> Self {
        Self {
            data,
            name,
            metric_descriptor: None,
        }
    }

    /// Creates a source with values `0.0` through `(len - 1) as f32`.
    ///
    /// # Examples
    /// ```ignore
    /// use chutoro_core::test_utils::NumericTestSource;
    ///
    /// let source = NumericTestSource::with_len("session-test", 3);
    /// assert_eq!(source.data(), &[0.0, 1.0, 2.0]);
    /// ```
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "f32 fixtures preserve Rust's integer-to-float rounding for large indices"
    )]
    pub(crate) fn with_len(name: &'static str, len: usize) -> Self {
        Self::new(name, (0..len).map(|value| value as f32).collect())
    }

    /// Creates a source with an explicit metric descriptor.
    ///
    /// # Examples
    /// ```ignore
    /// use chutoro_core::{MetricDescriptor, test_utils::NumericTestSource};
    ///
    /// let source = NumericTestSource::with_metric_descriptor(
    ///     "session-test",
    ///     vec![0.0, 1.0],
    ///     MetricDescriptor::new("session-test:abs"),
    /// );
    /// assert_eq!(source.metric_descriptor().as_str(), "session-test:abs");
    /// ```
    #[must_use]
    pub(crate) fn with_metric_descriptor(
        name: &'static str,
        data: Vec<f32>,
        metric_descriptor: MetricDescriptor,
    ) -> Self {
        Self {
            data,
            name,
            metric_descriptor: Some(metric_descriptor),
        }
    }

    /// Sets the metric descriptor while preserving this source's data and name.
    ///
    /// # Examples
    /// ```ignore
    /// use chutoro_core::{MetricDescriptor, test_utils::NumericTestSource};
    ///
    /// let source = NumericTestSource::with_len("session-test", 2)
    ///     .using_metric_descriptor(MetricDescriptor::new("session-test:abs"));
    /// assert_eq!(source.metric_descriptor().as_str(), "session-test:abs");
    /// ```
    #[must_use]
    pub(crate) fn using_metric_descriptor(mut self, metric_descriptor: MetricDescriptor) -> Self {
        self.metric_descriptor = Some(metric_descriptor);
        self
    }

    /// Returns the stored values for test assertions.
    ///
    /// # Examples
    /// ```ignore
    /// use chutoro_core::test_utils::NumericTestSource;
    ///
    /// let source = NumericTestSource::with_len("dummy", 2);
    /// assert_eq!(source.data(), &[0.0, 1.0]);
    /// ```
    #[must_use]
    pub(crate) fn data(&self) -> &[f32] {
        &self.data
    }
}

impl DataSource for NumericTestSource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        self.name
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        self.metric_descriptor.clone().unwrap_or_default()
    }

    #[expect(
        clippy::float_arithmetic,
        reason = "absolute difference is the required fixture distance semantics"
    )]
    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let a = self
            .data
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let b = self
            .data
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok((*a - *b).abs())
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
    source: NumericTestSource,
    calls: Arc<AtomicUsize>,
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
        Self {
            source: NumericTestSource::new(name, data),
            calls,
        }
    }

    /// Returns the backing distance counter for assertions.
    #[must_use]
    pub(crate) fn calls(&self) -> &Arc<AtomicUsize> {
        &self.calls
    }

    /// Returns an immutable view over the stored values.
    #[must_use]
    pub(crate) fn data(&self) -> &[f32] {
        self.source.data()
    }
}

impl DataSource for CountingSource {
    fn len(&self) -> usize {
        self.source.len()
    }

    fn name(&self) -> &str {
        self.source.name()
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.source.distance(left, right)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for shared proptest settings and data-source fixtures.

    use std::sync::atomic::Ordering;

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

    #[test]
    fn numeric_source_returns_absolute_difference() {
        let source = NumericTestSource::new("dummy", vec![1.0, 4.0]);

        assert_eq!(
            source
                .distance(0, 1)
                .expect("distance is in bounds")
                .to_bits(),
            3.0_f32.to_bits()
        );
    }

    #[rstest::rstest]
    #[case(3, 0, 3)]
    #[case(0, 3, 3)]
    fn numeric_source_reports_the_out_of_bounds_index(
        #[case] left: usize,
        #[case] right: usize,
        #[case] expected_index: usize,
    ) {
        let source = NumericTestSource::new("dummy", vec![0.0, 1.0, 2.0]);

        assert_eq!(
            source.distance(left, right),
            Err(DataSourceError::OutOfBounds {
                index: expected_index
            })
        );
    }

    #[test]
    fn numeric_source_exposes_its_configured_name_and_length() {
        let source = NumericTestSource::new("named", vec![0.0, 1.0, 2.0]);

        assert_eq!(source.name(), "named");
        assert_eq!(source.len(), 3);
    }

    #[test]
    fn range_constructor_produces_the_expected_values() {
        let source = NumericTestSource::with_len("session-test", 3);

        assert_eq!(
            source
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0.0_f32.to_bits(), 1.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
    }

    #[test]
    fn metric_descriptor_constructor_overrides_the_default() {
        let source = NumericTestSource::with_metric_descriptor(
            "session-test",
            vec![0.0, 1.0],
            MetricDescriptor::new("session-test:abs"),
        );

        assert_eq!(source.metric_descriptor().as_str(), "session-test:abs");
    }

    #[test]
    fn counting_source_increments_once_per_distance_call() {
        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingSource::new(vec![0.0, 2.0], Arc::clone(&calls));

        assert_eq!(
            source
                .distance(0, 1)
                .expect("distance is in bounds")
                .to_bits(),
            2.0_f32.to_bits()
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
