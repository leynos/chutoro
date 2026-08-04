//! Coverage-aware budget selection for HNSW property tests.

use chutoro_test_support::ci::property_test_profile::ProptestRunProfile;

use crate::hnsw::tests::support::is_coverage_job;

pub(crate) use super::budget_types::{
    InvalidTestCasesError, ShrinkIterations, StackSize, TestCases,
};

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
/// exceeded the 600s nextest timeout in CI. `MAX_IDEMPOTENCY_CASES` is the
/// Standard-job analogue for non-coverage runs.
const COVERAGE_IDEMPOTENCY_CASES: u32 = 1;
/// Coverage jobs cap shrink iterations to prevent long minimization tails.
/// Reduced from 32 to 8 because shrinking is rarely exercised for the
/// idempotency property and coverage instrumentation amplifies per-iteration
/// cost.
const COVERAGE_IDEMPOTENCY_MAX_SHRINK_ITERS: u32 = 8;
/// Non-coverage jobs keep deeper shrinking for better counterexample reduction.
const DEFAULT_IDEMPOTENCY_MAX_SHRINK_ITERS: u32 = 1024;
/// Default idempotency case count when no profile override is configured.
const DEFAULT_IDEMPOTENCY_CASES: u32 = 16;
/// Maximum idempotency case count for non-coverage (standard) runs.
///
/// Each idempotency case performs a full `CpuHnsw::build` and a complete
/// graph snapshot, making it substantially more expensive per iteration than
/// mutation or search cases. Cap at the default to ensure the test stays
/// within the 600-second nextest CI budget regardless of the `PROPTEST_CASES`
/// environment variable.
const MAX_IDEMPOTENCY_CASES: u32 = DEFAULT_IDEMPOTENCY_CASES;

/// Coverage jobs use fewer mutation cases to stay within CI time budgets.
/// Mutation tests build full HNSW indices per case, making each iteration
/// expensive under coverage instrumentation.
const COVERAGE_MUTATION_CASES: u32 = 4;
/// Coverage jobs cap mutation shrink iterations to prevent long minimization
/// tails under instrumentation overhead.
const COVERAGE_MUTATION_MAX_SHRINK_ITERS: u32 = 64;
/// Non-coverage jobs keep deeper shrinking for better counterexample reduction.
const DEFAULT_MUTATION_MAX_SHRINK_ITERS: u32 = 1024;
/// Default mutation case count when no profile override is configured.
const DEFAULT_MUTATION_CASES: u32 = 64;
/// Maximum mutation case count for non-forked standard runs.
///
/// Mutation cases can rebuild and revalidate the graph after up to twelve
/// add/delete/reconfigure steps. Cap pull request runs at the default budget
/// so `PROPTEST_CASES=250` cannot consume the 600-second nextest allowance,
/// while still allowing forked scheduled runs to request deeper coverage.
const MAX_MUTATION_CASES: u32 = DEFAULT_MUTATION_CASES;
/// Coverage jobs use fewer search cases to stay within CI time budgets.
/// Search correctness builds full HNSW indices and brute-force baselines per
/// case, so coverage instrumentation amplifies the cost substantially.
const COVERAGE_SEARCH_CASES: u32 = 4;
/// Coverage jobs cap search shrink iterations to avoid long minimization tails
/// under instrumentation overhead.
const COVERAGE_SEARCH_MAX_SHRINK_ITERS: u32 = 64;
/// Non-coverage jobs keep deeper shrinking for better counterexample reduction.
const DEFAULT_SEARCH_MAX_SHRINK_ITERS: u32 = 1024;
/// Default search case count when no profile override is configured.
const DEFAULT_SEARCH_CASES: u32 = 64;

fn select_cases(
    job: JobKind,
    configured: TestCases,
    coverage_cases: u32,
) -> Result<TestCases, InvalidTestCasesError> {
    if job.is_coverage() {
        TestCases::try_new(coverage_cases)
    } else {
        Ok(configured)
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

fn configured_cases(default_cases: u32) -> Result<TestCases, InvalidTestCasesError> {
    TestCases::try_new(property_run_profile(default_cases).cases())
}

/// Selects the number of idempotency test cases based on job kind.
///
/// Returns capped configured cases for non-coverage jobs, or a reduced count
/// (`COVERAGE_IDEMPOTENCY_CASES`) for coverage-instrumented runs.
///
/// # Errors
/// Returns `InvalidTestCasesError` if the selected case count is zero.
pub(crate) fn select_idempotency_cases(
    job: JobKind,
    configured: TestCases,
) -> Result<TestCases, InvalidTestCasesError> {
    if job.is_coverage() {
        TestCases::try_new(COVERAGE_IDEMPOTENCY_CASES)
    } else {
        TestCases::try_new(configured.get().min(MAX_IDEMPOTENCY_CASES))
    }
}

/// Returns the number of idempotency test cases, auto-detecting the job kind.
///
/// Calls `select_idempotency_cases` with the detected `JobKind` and
/// the default idempotency case count from the property test profile.
///
/// # Errors
/// Returns `InvalidTestCasesError` if the configured case count is zero.
pub(crate) fn idempotency_cases() -> Result<TestCases, InvalidTestCasesError> {
    select_idempotency_cases(
        JobKind::detect(),
        configured_cases(DEFAULT_IDEMPOTENCY_CASES)?,
    )
}

/// Selects the maximum shrink iterations for idempotency tests based on job kind.
///
/// Returns a reduced iteration count for coverage jobs to avoid long minimization
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
/// Returns capped configured cases for non-forked non-coverage jobs, or a
/// reduced count (`COVERAGE_MUTATION_CASES`) for coverage-instrumented runs.
///
/// # Errors
/// Returns `InvalidTestCasesError` if the selected case count is zero.
pub(crate) fn select_mutation_cases(
    job: JobKind,
    configured: TestCases,
) -> Result<TestCases, InvalidTestCasesError> {
    select_mutation_cases_for_fork(job, configured, false)
}

/// Selects mutation case count while preserving forked deep-run budgets.
///
/// # Errors
/// Returns `InvalidTestCasesError` if the selected case count is zero.
pub(crate) fn select_mutation_cases_for_fork(
    job: JobKind,
    configured: TestCases,
    fork: bool,
) -> Result<TestCases, InvalidTestCasesError> {
    if job.is_coverage() {
        TestCases::try_new(COVERAGE_MUTATION_CASES)
    } else if fork {
        Ok(configured)
    } else {
        TestCases::try_new(configured.get().min(MAX_MUTATION_CASES))
    }
}

/// Returns the number of mutation test cases, auto-detecting the job kind.
///
/// Calls `select_mutation_cases_for_fork` with the detected `JobKind` and the
/// mutation case count from the property test profile.
///
/// # Errors
/// Returns `InvalidTestCasesError` if the configured case count is zero.
pub(crate) fn mutation_cases() -> Result<TestCases, InvalidTestCasesError> {
    let profile = property_run_profile(DEFAULT_MUTATION_CASES);
    select_mutation_cases_for_fork(
        JobKind::detect(),
        TestCases::try_new(profile.cases())?,
        profile.fork(),
    )
}

/// Selects the maximum shrink iterations for mutation tests based on job kind.
///
/// Returns a reduced iteration count for coverage jobs to prevent long minimization
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
///
/// # Errors
/// Returns `InvalidTestCasesError` if the selected case count is zero.
pub(crate) fn select_search_cases(
    job: JobKind,
    configured: TestCases,
) -> Result<TestCases, InvalidTestCasesError> {
    select_cases(job, configured, COVERAGE_SEARCH_CASES)
}

/// Returns the number of search property test cases, auto-detecting the job kind.
///
/// Calls `select_search_cases` with the detected `JobKind` and
/// the default search case count from the property test profile.
///
/// # Errors
/// Returns `InvalidTestCasesError` if the configured case count is zero.
pub(crate) fn search_cases() -> Result<TestCases, InvalidTestCasesError> {
    select_search_cases(JobKind::detect(), configured_cases(DEFAULT_SEARCH_CASES)?)
}

/// Selects the maximum shrink iterations for search property tests based on job kind.
///
/// Returns a reduced iteration count for coverage jobs to avoid long minimization
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
