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

/// Coverage jobs use fewer idempotency cases to stay within CI time budgets.
const COVERAGE_IDEMPOTENCY_CASES: u32 = 4;
/// Coverage jobs cap shrink iterations to prevent long minimization tails.
const COVERAGE_IDEMPOTENCY_MAX_SHRINK_ITERS: u32 = 128;
/// Non-coverage jobs keep deeper shrinking for better counterexample reduction.
const DEFAULT_IDEMPOTENCY_MAX_SHRINK_ITERS: u32 = 1024;
/// Default idempotency case count when no profile override is configured.
const DEFAULT_IDEMPOTENCY_CASES: u32 = 16;

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
    cases: u32,
    max_shrink_iters: u32,
    stack_size: usize,
) -> TestCaseResult {
    run_test_with_profile(
        cases,
        max_shrink_iters,
        stack_size,
        run_mutation_proptest_with_stack,
    )
}

/// Runs a search property test with custom configuration parameters.
pub(super) fn run_search_test(cases: u32, max_shrink_iters: u32) -> TestCaseResult {
    run_test_with_profile_no_stack(cases, max_shrink_iters, run_search_proptest)
}

/// Runs an idempotency property test with custom configuration parameters.
pub(super) fn run_idempotency_test(
    cases: u32,
    max_shrink_iters: u32,
    stack_size: usize,
) -> TestCaseResult {
    run_test_with_profile(
        cases,
        max_shrink_iters,
        stack_size,
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

fn is_coverage_run() -> bool {
    cfg!(coverage)
}

fn is_coverage_env_run() -> bool {
    std::env::var_os("LLVM_PROFILE_FILE").is_some() || std::env::var_os("CARGO_LLVM_COV").is_some()
}

fn is_coverage_job() -> bool {
    is_coverage_run() || is_coverage_env_run()
}

fn property_run_profile(default_cases: u32) -> ProptestRunProfile {
    ProptestRunProfile::load(default_cases, false)
}
