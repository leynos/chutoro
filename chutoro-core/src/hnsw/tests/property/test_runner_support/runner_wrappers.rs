//! Proptest execution wrappers for HNSW property tests.

use proptest::{
    prelude::any,
    test_runner::{Config, TestCaseError, TestCaseResult, TestError, TestRunner},
};

use super::budget_selection::{ShrinkIterations, StackSize, TestCases, property_run_profile};
use crate::hnsw::tests::property::{
    idempotency_property::run_idempotency_property,
    mutation_property::run_mutation_property,
    search_property::run_search_correctness_property,
    strategies::{hnsw_fixture_strategy, idempotency_plan_strategy, mutation_plan_strategy},
};

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
pub(crate) fn run_mutation_test(
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
pub(crate) fn run_search_test(
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
) -> TestCaseResult {
    run_test_with_profile_no_stack(cases, max_shrink_iters, run_search_proptest)
}

/// Runs an idempotency property test with custom configuration parameters.
pub(crate) fn run_idempotency_test(
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
    let handle = std::thread::Builder::new()
        .name("hnsw-mutation".into())
        .stack_size(stack_size)
        .spawn(move || run_mutation_proptest(config))
        .map_err(|e| {
            TestCaseError::fail(format!(
                "failed to spawn mutation runner thread: {e}"
            ))
        })?;

    handle.join().map_err(|panic_payload| {
        TestCaseError::fail(format!(
            "mutation runner panicked: {panic_payload:?}"
        ))
    })?
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
    let handle = std::thread::Builder::new()
        .name("hnsw-idempotency".into())
        .stack_size(stack_size)
        .spawn(move || run_idempotency_proptest(config))
        .map_err(|e| {
            TestCaseError::fail(format!(
                "failed to spawn idempotency runner thread: {e}"
            ))
        })?;

    handle.join().map_err(|panic_payload| {
        TestCaseError::fail(format!(
            "idempotency runner panicked: {panic_payload:?}"
        ))
    })?
}

fn run_idempotency_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), idempotency_plan_strategy()),
        "hnsw idempotency proptest",
        |(fixture, plan)| run_idempotency_property(fixture, plan),
    )
}
