//! Shared proptest runner and coverage-budget helpers for HNSW property tests.

use chutoro_test_support::ci::property_test_profile::ProptestRunProfile;
use proptest::{
    prelude::any,
    test_runner::{Config, TestCaseError, TestCaseResult, TestError, TestRunner},
};

use crate::hnsw::tests::support::is_coverage_job;

use super::{
    idempotency_property::run_idempotency_property,
    mutation_property::run_mutation_property,
    search_property::run_search_correctness_property,
    strategies::{hnsw_fixture_strategy, idempotency_plan_strategy, mutation_plan_strategy},
};

/// Number of test cases to execute in a property test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TestCases(u32);

impl TestCases {
    /// Creates a new TestCases value.
    ///
    /// # Panics
    /// Panics if `cases` is zero (at least one test case is required).
    pub(super) fn new(cases: u32) -> Self {
        assert!(cases > 0, "test cases must be > 0");
        Self(cases)
    }

    pub(super) fn get(self) -> u32 {
        self.0
    }
}

impl From<TestCases> for u32 {
    fn from(cases: TestCases) -> u32 {
        cases.0
    }
}

/// Maximum number of shrinking iterations to attempt when minimising a failing test case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ShrinkIterations(u32);

impl ShrinkIterations {
    pub(super) fn new(iterations: u32) -> Self {
        Self(iterations)
    }

    pub(super) fn get(self) -> u32 {
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
pub(super) struct StackSize(usize);

impl StackSize {
    /// Minimum safe stack size (1 MiB).
    const MIN_STACK_SIZE: usize = 1024 * 1024;

    pub(super) fn new(size: usize) -> Self {
        assert!(
            size >= Self::MIN_STACK_SIZE,
            "stack size must be >= {} bytes",
            Self::MIN_STACK_SIZE
        );
        Self(size)
    }

    pub(super) fn get(self) -> usize {
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
pub(super) enum JobKind {
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
/// Coverage jobs cap search shrink iterations to avoid long minimization tails
/// under instrumentation overhead.
const COVERAGE_SEARCH_MAX_SHRINK_ITERS: u32 = 64;
/// Non-coverage jobs keep deeper shrinking for better counterexample reduction.
const DEFAULT_SEARCH_MAX_SHRINK_ITERS: u32 = 1024;
/// Default search case count when no profile override is configured.
const DEFAULT_SEARCH_CASES: u32 = 64;

fn run_test_with_profile<F>(
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
    test_runner: F,
) -> TestCaseResult
where
    F: FnOnce(Config, usize) -> TestCaseResult,
{
    let profile = property_run_profile(cases.get());
    let config = PropertyRunnerConfig {
        cases: TestCases::new(profile.cases()),
        fork: profile.fork(),
        max_shrink_iters,
        stack_size,
    };
    run_test_with_config(config, test_runner)
}

fn run_test_with_profile_no_stack<F>(
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    test_runner: F,
) -> TestCaseResult
where
    F: FnOnce(Config) -> TestCaseResult,
{
    let profile = property_run_profile(cases.get());
    test_runner(Config {
        cases: profile.cases(),
        fork: profile.fork(),
        max_shrink_iters: max_shrink_iters.get(),
        ..Config::default()
    })
}

/// Runs a mutation property test with custom configuration parameters.
pub(super) fn run_mutation_test(
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
) -> TestCaseResult {
    run_test_with_profile(
        cases,
        max_shrink_iters,
        stack_size,
        run_mutation_proptest_with_stack,
    )
}

/// Runs a search property test with custom configuration parameters.
pub(super) fn run_search_test(
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
) -> TestCaseResult {
    run_test_with_profile_no_stack(cases, max_shrink_iters, run_search_proptest)
}

/// Runs an idempotency property test with custom configuration parameters.
pub(super) fn run_idempotency_test(
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
) -> TestCaseResult {
    run_test_with_profile(
        cases,
        max_shrink_iters,
        stack_size,
        run_idempotency_proptest_with_stack,
    )
}

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

pub(super) fn select_idempotency_cases(job: JobKind, configured: TestCases) -> TestCases {
    select_cases(job, configured, COVERAGE_IDEMPOTENCY_CASES)
}

pub(super) fn idempotency_cases() -> TestCases {
    select_idempotency_cases(
        JobKind::detect(),
        configured_cases(DEFAULT_IDEMPOTENCY_CASES),
    )
}

pub(super) fn select_idempotency_shrink_iters(job: JobKind) -> ShrinkIterations {
    select_shrink_iterations(
        job,
        COVERAGE_IDEMPOTENCY_MAX_SHRINK_ITERS,
        DEFAULT_IDEMPOTENCY_MAX_SHRINK_ITERS,
    )
}

pub(super) fn idempotency_shrink_iters() -> ShrinkIterations {
    select_idempotency_shrink_iters(JobKind::detect())
}

pub(super) fn select_mutation_cases(job: JobKind, configured: TestCases) -> TestCases {
    select_cases(job, configured, COVERAGE_MUTATION_CASES)
}

pub(super) fn mutation_cases() -> TestCases {
    select_mutation_cases(JobKind::detect(), configured_cases(DEFAULT_MUTATION_CASES))
}

pub(super) fn select_mutation_shrink_iters(job: JobKind) -> ShrinkIterations {
    select_shrink_iterations(
        job,
        COVERAGE_MUTATION_MAX_SHRINK_ITERS,
        DEFAULT_MUTATION_MAX_SHRINK_ITERS,
    )
}

pub(super) fn mutation_shrink_iters() -> ShrinkIterations {
    select_mutation_shrink_iters(JobKind::detect())
}

pub(super) fn select_search_cases(job: JobKind, configured: TestCases) -> TestCases {
    select_cases(job, configured, COVERAGE_SEARCH_CASES)
}

pub(super) fn search_cases() -> TestCases {
    select_search_cases(JobKind::detect(), configured_cases(DEFAULT_SEARCH_CASES))
}

pub(super) fn select_search_shrink_iters(job: JobKind) -> ShrinkIterations {
    select_shrink_iterations(
        job,
        COVERAGE_SEARCH_MAX_SHRINK_ITERS,
        DEFAULT_SEARCH_MAX_SHRINK_ITERS,
    )
}

pub(super) fn search_shrink_iters() -> ShrinkIterations {
    select_search_shrink_iters(JobKind::detect())
}

/// Runs a property test with the given configuration and strategy.
fn run_proptest<S, F>(config: Config, strategy: S, test_name: &str, property: F) -> TestCaseResult
where
    S: proptest::strategy::Strategy,
    F: Fn(S::Value) -> TestCaseResult,
{
    let mut runner = TestRunner::new(config);
    runner
        .run(&strategy, property)
        .map_err(|err| map_test_error(err, test_name))
}

/// Maps TestError to TestCaseError with formatted messages.
fn map_test_error(err: TestError<impl std::fmt::Debug>, test_name: &str) -> TestCaseError {
    match err {
        TestError::Abort(reason) => TestCaseError::fail(format!("{test_name} aborted: {reason}")),
        TestError::Fail(reason, value) => TestCaseError::fail(format!(
            "{test_name} failed: {reason}; minimal input: {value:#?}"
        )),
    }
}

/// Runs a mutation property test with custom configuration and stack size.
fn run_mutation_proptest_with_stack(config: Config, stack_size: usize) -> TestCaseResult {
    std::thread::Builder::new()
        .name("hnsw-mutation".into())
        .stack_size(stack_size)
        .spawn(move || run_mutation_proptest(config))
        .expect("spawn mutation runner")
        .join()
        .expect("mutation runner panicked")
}

fn run_mutation_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), mutation_plan_strategy()),
        "hnsw mutation proptest",
        |(fixture, plan)| run_mutation_property(fixture, plan),
    )
}

fn run_search_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), any::<u16>(), any::<u16>()),
        "hnsw search proptest",
        |(fixture, query_hint, k_hint)| {
            run_search_correctness_property(fixture, query_hint, k_hint)
        },
    )
}

/// Configuration for property runners that execute within dedicated threads.
#[derive(Clone, Copy)]
struct PropertyRunnerConfig {
    cases: TestCases,
    fork: bool,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
}

/// Runs a property test with custom configuration parameters and stack size.
fn run_test_with_config<F>(runner_config: PropertyRunnerConfig, runner: F) -> TestCaseResult
where
    F: FnOnce(Config, usize) -> TestCaseResult,
{
    runner(
        Config {
            cases: runner_config.cases.get(),
            fork: runner_config.fork,
            max_shrink_iters: runner_config.max_shrink_iters.get(),
            ..Config::default()
        },
        runner_config.stack_size.get(),
    )
}

/// Runs an idempotency property test with custom configuration and stack size.
fn run_idempotency_proptest_with_stack(config: Config, stack_size: usize) -> TestCaseResult {
    std::thread::Builder::new()
        .name("hnsw-idempotency".into())
        .stack_size(stack_size)
        .spawn(move || run_idempotency_proptest(config))
        .expect("spawn idempotency runner")
        .join()
        .expect("idempotency runner panicked")
}

fn run_idempotency_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), idempotency_plan_strategy()),
        "hnsw idempotency proptest",
        |(fixture, plan)| run_idempotency_property(fixture, plan),
    )
}

/// Loads the proptest run profile for the supplied default case count.
fn property_run_profile(default_cases: u32) -> ProptestRunProfile {
    ProptestRunProfile::load(default_cases, false)
}
