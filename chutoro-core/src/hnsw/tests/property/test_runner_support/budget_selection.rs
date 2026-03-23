//! Coverage-aware budget selection for HNSW property tests.

use chutoro_test_support::ci::property_test_profile::ProptestRunProfile;

use crate::hnsw::tests::support::is_coverage_job;

/// Number of test cases to execute in a property test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TestCases(u32);

/// Error returned when test case count is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvalidTestCasesError;

impl std::fmt::Display for InvalidTestCasesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "test cases must be > 0")
    }
}

impl std::error::Error for InvalidTestCasesError {}

impl TestCases {
    /// Creates a new TestCases value, returning an error if invalid.
    ///
    /// # Errors
    /// Returns `InvalidTestCasesError` if `cases` is zero.
    pub(crate) fn try_new(cases: u32) -> Result<Self, InvalidTestCasesError> {
        if cases > 0 {
            Ok(Self(cases))
        } else {
            Err(InvalidTestCasesError)
        }
    }

    /// Creates a new TestCases value.
    ///
    /// # Panics
    /// Panics if `cases` is zero (at least one test case is required).
    pub(crate) fn new(cases: u32) -> Self {
        Self::try_new(cases).expect("test cases must be > 0")
    }

    /// Returns the number of test cases.
    pub(crate) fn get(self) -> u32 {
        self.0
    }
}

impl From<TestCases> for u32 {
    fn from(cases: TestCases) -> u32 {
        cases.0
    }
}

/// Maximum number of shrinking iterations to attempt when minimising a failing test case.
///
/// A value of 0 disables shrinking entirely. In practice, positive values are used to enable
/// counterexample minimisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShrinkIterations(u32);

impl ShrinkIterations {
    /// Creates a new ShrinkIterations value.
    ///
    /// Setting `iterations` to 0 disables shrinking.
    pub(crate) fn new(iterations: u32) -> Self {
        Self(iterations)
    }

    /// Returns the number of shrink iterations.
    ///
    /// A return value of 0 means shrinking is disabled.
    pub(crate) fn get(self) -> u32 {
        self.0
    }
}

impl From<ShrinkIterations> for u32 {
    fn from(iters: ShrinkIterations) -> u32 {
        iters.0
    }
}

/// Stack size in bytes for property test runner threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StackSize(usize);

/// Error returned when stack size is below the minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvalidStackSizeError {
    provided: usize,
    minimum: usize,
}

impl std::fmt::Display for InvalidStackSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stack size must be >= {} bytes, got {}",
            self.minimum, self.provided
        )
    }
}

impl std::error::Error for InvalidStackSizeError {}

impl StackSize {
    /// Minimum safe stack size (1 MiB).
    const MIN_STACK_SIZE: usize = 1024 * 1024;

    /// Creates a new StackSize value, returning an error if below minimum.
    ///
    /// # Errors
    /// Returns `InvalidStackSizeError` if `size` is below `MIN_STACK_SIZE`.
    pub(crate) fn try_new(size: usize) -> Result<Self, InvalidStackSizeError> {
        if size >= Self::MIN_STACK_SIZE {
            Ok(Self(size))
        } else {
            Err(InvalidStackSizeError {
                provided: size,
                minimum: Self::MIN_STACK_SIZE,
            })
        }
    }

    /// Creates a new StackSize value.
    ///
    /// # Panics
    /// Panics if `size` is below `MIN_STACK_SIZE`.
    pub(crate) fn new(size: usize) -> Self {
        Self::try_new(size).expect("stack size must be >= minimum")
    }

    /// Returns the stack size in bytes.
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

impl From<StackSize> for usize {
    fn from(size: StackSize) -> usize {
        size.0
    }
}

/// Whether the current job runs under coverage instrumentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobKind {
    Coverage,
    Standard,
}

impl JobKind {
    fn detect() -> Self {
        if is_coverage_job() {
            Self::Coverage
        } else {
            Self::Standard
        }
    }

    fn is_coverage(self) -> bool {
        self == Self::Coverage
    }
}

/// Coverage jobs use fewer idempotency cases to stay within CI time budgets.
/// Reduced from 2 to 1 after coverage-instrumented builds occasionally
/// exceeded the 600s nextest timeout in CI.
const COVERAGE_IDEMPOTENCY_CASES: u32 = 1;
/// Coverage jobs cap shrink iterations to prevent long minimisation tails.
/// Reduced from 32 to 8 because shrinking is rarely exercised for the
/// idempotency property and coverage instrumentation amplifies per-iteration
/// cost.
const COVERAGE_IDEMPOTENCY_MAX_SHRINK_ITERS: u32 = 8;
/// Non-coverage jobs keep deeper shrinking for better counterexample reduction.
const DEFAULT_IDEMPOTENCY_MAX_SHRINK_ITERS: u32 = 1024;
/// Default idempotency case count when no profile override is configured.
const DEFAULT_IDEMPOTENCY_CASES: u32 = 16;

/// Coverage jobs use fewer mutation cases to stay within CI time budgets.
/// Mutation tests build full HNSW indices per case, making each iteration
/// expensive under coverage instrumentation.
const COVERAGE_MUTATION_CASES: u32 = 4;
/// Coverage jobs cap mutation shrink iterations to prevent long minimisation
/// tails under instrumentation overhead.
const COVERAGE_MUTATION_MAX_SHRINK_ITERS: u32 = 64;
/// Non-coverage jobs keep deeper shrinking for better counterexample reduction.
const DEFAULT_MUTATION_MAX_SHRINK_ITERS: u32 = 1024;
/// Default mutation case count when no profile override is configured.
const DEFAULT_MUTATION_CASES: u32 = 64;
/// Coverage jobs use fewer search cases to stay within CI time budgets.
/// Search correctness builds full HNSW indices and brute-force baselines per
/// case, so coverage instrumentation amplifies the cost substantially.
const COVERAGE_SEARCH_CASES: u32 = 4;
/// Coverage jobs cap search shrink iterations to avoid long minimisation tails
/// under instrumentation overhead.
const COVERAGE_SEARCH_MAX_SHRINK_ITERS: u32 = 64;
/// Non-coverage jobs keep deeper shrinking for better counterexample reduction.
const DEFAULT_SEARCH_MAX_SHRINK_ITERS: u32 = 1024;
/// Default search case count when no profile override is configured.
const DEFAULT_SEARCH_CASES: u32 = 64;

fn select_cases(job: JobKind, configured: TestCases, coverage_cases: u32) -> TestCases {
    if job.is_coverage() {
        TestCases::new(coverage_cases)
    } else {
        configured
    }
}

fn select_shrink_iterations(
    job: JobKind,
    coverage_iters: u32,
    default_iters: u32,
) -> ShrinkIterations {
    if job.is_coverage() {
        ShrinkIterations::new(coverage_iters)
    } else {
        ShrinkIterations::new(default_iters)
    }
}

fn configured_cases(default_cases: u32) -> TestCases {
    TestCases::new(property_run_profile(default_cases).cases())
}

/// Selects the number of idempotency test cases based on job kind.
///
/// Returns the configured cases for non-coverage jobs, or a reduced count
/// (`COVERAGE_IDEMPOTENCY_CASES`) for coverage-instrumented runs.
pub(crate) fn select_idempotency_cases(job: JobKind, configured: TestCases) -> TestCases {
    select_cases(job, configured, COVERAGE_IDEMPOTENCY_CASES)
}

/// Returns the number of idempotency test cases, auto-detecting the job kind.
///
/// Calls `select_idempotency_cases` with the detected `JobKind` and
/// the default idempotency case count from the property test profile.
pub(crate) fn idempotency_cases() -> TestCases {
    select_idempotency_cases(
        JobKind::detect(),
        configured_cases(DEFAULT_IDEMPOTENCY_CASES),
    )
}

/// Selects the maximum shrink iterations for idempotency tests based on job kind.
///
/// Returns a reduced iteration count for coverage jobs to avoid long minimisation
/// tails under instrumentation overhead.
pub(crate) fn select_idempotency_shrink_iters(job: JobKind) -> ShrinkIterations {
    select_shrink_iterations(
        job,
        COVERAGE_IDEMPOTENCY_MAX_SHRINK_ITERS,
        DEFAULT_IDEMPOTENCY_MAX_SHRINK_ITERS,
    )
}

/// Returns the maximum shrink iterations for idempotency tests, auto-detecting the job kind.
///
/// Calls `select_idempotency_shrink_iters` with the detected `JobKind`.
pub(crate) fn idempotency_shrink_iters() -> ShrinkIterations {
    select_idempotency_shrink_iters(JobKind::detect())
}

/// Selects the number of mutation test cases based on job kind.
///
/// Returns the configured cases for non-coverage jobs, or a reduced count
/// (`COVERAGE_MUTATION_CASES`) for coverage-instrumented runs.
pub(crate) fn select_mutation_cases(job: JobKind, configured: TestCases) -> TestCases {
    select_cases(job, configured, COVERAGE_MUTATION_CASES)
}

/// Returns the number of mutation test cases, auto-detecting the job kind.
///
/// Calls `select_mutation_cases` with the detected `JobKind` and
/// the default mutation case count from the property test profile.
pub(crate) fn mutation_cases() -> TestCases {
    select_mutation_cases(JobKind::detect(), configured_cases(DEFAULT_MUTATION_CASES))
}

/// Selects the maximum shrink iterations for mutation tests based on job kind.
///
/// Returns a reduced iteration count for coverage jobs to prevent long minimisation
/// tails under instrumentation overhead.
pub(crate) fn select_mutation_shrink_iters(job: JobKind) -> ShrinkIterations {
    select_shrink_iterations(
        job,
        COVERAGE_MUTATION_MAX_SHRINK_ITERS,
        DEFAULT_MUTATION_MAX_SHRINK_ITERS,
    )
}

/// Returns the maximum shrink iterations for mutation tests, auto-detecting the job kind.
///
/// Calls `select_mutation_shrink_iters` with the detected `JobKind`.
pub(crate) fn mutation_shrink_iters() -> ShrinkIterations {
    select_mutation_shrink_iters(JobKind::detect())
}

/// Selects the number of search property test cases based on job kind.
///
/// Returns the configured cases for non-coverage jobs, or a reduced count
/// (`COVERAGE_SEARCH_CASES`) for coverage-instrumented runs.
pub(crate) fn select_search_cases(job: JobKind, configured: TestCases) -> TestCases {
    select_cases(job, configured, COVERAGE_SEARCH_CASES)
}

/// Returns the number of search property test cases, auto-detecting the job kind.
///
/// Calls `select_search_cases` with the detected `JobKind` and
/// the default search case count from the property test profile.
pub(crate) fn search_cases() -> TestCases {
    select_search_cases(JobKind::detect(), configured_cases(DEFAULT_SEARCH_CASES))
}

/// Selects the maximum shrink iterations for search property tests based on job kind.
///
/// Returns a reduced iteration count for coverage jobs to avoid long minimisation
/// tails under instrumentation overhead.
pub(crate) fn select_search_shrink_iters(job: JobKind) -> ShrinkIterations {
    select_shrink_iterations(
        job,
        COVERAGE_SEARCH_MAX_SHRINK_ITERS,
        DEFAULT_SEARCH_MAX_SHRINK_ITERS,
    )
}

/// Returns the maximum shrink iterations for search property tests, auto-detecting the job kind.
///
/// Calls `select_search_shrink_iters` with the detected `JobKind`.
pub(crate) fn search_shrink_iters() -> ShrinkIterations {
    select_search_shrink_iters(JobKind::detect())
}

/// Loads the proptest run profile for the given default case count.
///
/// Returns a `ProptestRunProfile` loaded from environment overrides with
/// `fork` disabled, allowing the profile to adjust case counts based on CI configuration.
pub(crate) fn property_run_profile(default_cases: u32) -> ProptestRunProfile {
    ProptestRunProfile::load(default_cases, false)
}
