//! Shared proptest runner and coverage-budget helpers for HNSW property tests.

use chutoro_test_support::ci::property_test_profile::ProptestRunProfile;
use proptest::{
    prelude::any,
    test_runner::{Config, TestCaseError, TestCaseResult, TestError, TestRunner},
};

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

/// Coverage jobs use fewer idempotency cases to stay within CI time budgets.
/// Reduced from 4 to 2 after coverage-instrumented builds exceeded the 180s
/// nextest timeout in CI (see PR #89).
const COVERAGE_IDEMPOTENCY_CASES: u32 = 2;
/// Coverage jobs cap shrink iterations to prevent long minimisation tails.
/// Reduced from 128 to 32 because shrinking is rarely exercised for the
/// idempotency property and coverage instrumentation amplifies per-iteration
/// cost.
const COVERAGE_IDEMPOTENCY_MAX_SHRINK_ITERS: u32 = 32;
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

fn run_test_with_profile<F>(
    cases: u32,
    max_shrink_iters: u32,
    stack_size: usize,
    test_runner: F,
) -> TestCaseResult
where
    F: FnOnce(Config, usize) -> TestCaseResult,
{
    let profile = property_run_profile(cases);
    let config = PropertyRunnerConfig {
        cases: profile.cases(),
        fork: profile.fork(),
        max_shrink_iters,
        stack_size,
    };
    run_test_with_config(config, test_runner)
}

fn run_test_with_profile_no_stack<F>(
    cases: u32,
    max_shrink_iters: u32,
    test_runner: F,
) -> TestCaseResult
where
    F: FnOnce(Config) -> TestCaseResult,
{
    let profile = property_run_profile(cases);
    test_runner(Config {
        cases: profile.cases(),
        fork: profile.fork(),
        max_shrink_iters,
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
        cases.get(),
        max_shrink_iters.get(),
        stack_size.get(),
        run_mutation_proptest_with_stack,
    )
}

/// Runs a search property test with custom configuration parameters.
pub(super) fn run_search_test(
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
) -> TestCaseResult {
    run_test_with_profile_no_stack(cases.get(), max_shrink_iters.get(), run_search_proptest)
}

/// Runs an idempotency property test with custom configuration parameters.
pub(super) fn run_idempotency_test(
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
) -> TestCaseResult {
    run_test_with_profile(
        cases.get(),
        max_shrink_iters.get(),
        stack_size.get(),
        run_idempotency_proptest_with_stack,
    )
}

/// Selects the idempotency case count for coverage and non-coverage jobs.
pub(super) fn select_idempotency_cases(is_coverage_job: bool, configured_cases: u32) -> u32 {
    if is_coverage_job {
        COVERAGE_IDEMPOTENCY_CASES
    } else {
        configured_cases
    }
}

/// Returns the idempotency case count using runtime job detection.
pub(super) fn idempotency_cases() -> u32 {
    let configured_cases = property_run_profile(DEFAULT_IDEMPOTENCY_CASES).cases();
    select_idempotency_cases(is_coverage_job(), configured_cases)
}

/// Selects the max shrink iterations for coverage and non-coverage jobs.
pub(super) fn select_idempotency_shrink_iters(is_coverage_job: bool) -> u32 {
    if is_coverage_job {
        COVERAGE_IDEMPOTENCY_MAX_SHRINK_ITERS
    } else {
        DEFAULT_IDEMPOTENCY_MAX_SHRINK_ITERS
    }
}

/// Returns idempotency shrink iterations using runtime job detection.
pub(super) fn idempotency_shrink_iters() -> u32 {
    select_idempotency_shrink_iters(is_coverage_job())
}

/// Selects the mutation case count for coverage and non-coverage jobs.
pub(super) fn select_mutation_cases(is_coverage_job: bool, configured_cases: u32) -> u32 {
    if is_coverage_job {
        COVERAGE_MUTATION_CASES
    } else {
        configured_cases
    }
}

/// Returns the mutation case count using runtime job detection.
pub(super) fn mutation_cases() -> u32 {
    let configured_cases = property_run_profile(DEFAULT_MUTATION_CASES).cases();
    select_mutation_cases(is_coverage_job(), configured_cases)
}

/// Selects the max shrink iterations for mutation coverage and non-coverage jobs.
pub(super) fn select_mutation_shrink_iters(is_coverage_job: bool) -> u32 {
    if is_coverage_job {
        COVERAGE_MUTATION_MAX_SHRINK_ITERS
    } else {
        DEFAULT_MUTATION_MAX_SHRINK_ITERS
    }
}

/// Returns mutation shrink iterations using runtime job detection.
pub(super) fn mutation_shrink_iters() -> u32 {
    select_mutation_shrink_iters(is_coverage_job())
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
    cases: u32,
    fork: bool,
    max_shrink_iters: u32,
    stack_size: usize,
}

/// Runs a property test with custom configuration parameters and stack size.
fn run_test_with_config<F>(runner_config: PropertyRunnerConfig, runner: F) -> TestCaseResult
where
    F: FnOnce(Config, usize) -> TestCaseResult,
{
    runner(
        Config {
            cases: runner_config.cases,
            fork: runner_config.fork,
            max_shrink_iters: runner_config.max_shrink_iters,
            ..Config::default()
        },
        runner_config.stack_size,
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

/// Returns `true` when code is compiled with coverage instrumentation enabled.
fn is_coverage_run() -> bool {
    cfg!(coverage)
}

/// Returns `true` when compile-time environment indicates an llvm-cov build.
///
/// Some runners sanitize runtime environments for test processes, so relying on
/// runtime vars alone can miss coverage context.
fn is_coverage_build_env() -> bool {
    option_env!("CARGO_LLVM_COV").is_some() || option_env!("LLVM_PROFILE_FILE").is_some()
}

/// Returns `true` when the runtime environment indicates a coverage collection run.
fn is_coverage_env_run() -> bool {
    std::env::var_os("LLVM_PROFILE_FILE").is_some() || std::env::var_os("CARGO_LLVM_COV").is_some()
}

/// Detects whether the current execution context should use coverage job budgets.
fn is_coverage_job() -> bool {
    is_coverage_run() || is_coverage_build_env() || is_coverage_env_run()
}

/// Loads the proptest run profile for the supplied default case count.
fn property_run_profile(default_cases: u32) -> ProptestRunProfile {
    ProptestRunProfile::load(default_cases, false)
}
