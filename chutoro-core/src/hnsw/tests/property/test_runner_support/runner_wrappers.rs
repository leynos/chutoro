//! Proptest execution wrappers for HNSW property tests.

use chutoro_test_support::ci::property_test_profile::PROPTEST_RNG_SEED;
use proptest::{
    prelude::any,
    test_runner::{Config, RngSeed, TestCaseError, TestCaseResult, TestError, TestRunner},
};

use super::budget_selection::{ShrinkIterations, StackSize, TestCases, property_run_profile};
use crate::hnsw::tests::property::{
    idempotency_property::run_idempotency_property,
    mutation_property::run_mutation_property,
    search_property::run_search_correctness_property,
    strategies::{hnsw_fixture_strategy, idempotency_plan_strategy, mutation_plan_strategy},
};

/// The libtest path of the `#[test]` a runner is executing.
///
/// proptest's fork mode re-executes this test binary and selects the case's
/// test with `--exact`, so `Config::test_name` has to be that test's own
/// path rather than a description of the suite. Supplying nothing aborts the
/// run before the first case, which is what the weekly lane did for months
/// (#260); supplying the wrong name selects no test at all, which is worse,
/// because the child then reports nothing rather than failing loudly.
/// The `forked_proptest!` macro derives the value from the test's own
/// identifier so the two cannot drift apart.
///
/// The leading crate name is stripped by `fix_module_path` in `rusty_fork`
/// before selection, so `module_path!()` is the right prefix to use here.
pub(crate) type TestPath = &'static str;

/// Defines a `#[test]` that drives a handbuilt proptest runner, supplying
/// the test's own libtest path as proptest's `test_name`.
///
/// proptest's fork mode re-executes this binary and selects the case's test
/// by that exact name. Deriving it from the function's own identifier is what
/// keeps the two from drifting apart: a written string would still compile
/// after a rename, and the weekly lane would go back to reporting nothing.
/// The generated test binds `$path` to the derived value for the body to
/// hand on.
///
/// The `#[test]` attribute is supplied here, so a use of this macro carries
/// only the attributes that vary, such as `#[ignore]`.
///
/// `a_forked_run_executes_at_least_one_case` is defined with this macro, so
/// the derivation is held up by a run that actually forks rather than by
/// inspection.
macro_rules! forked_proptest {
    ($(#[$attribute:meta])* fn $name:ident($path:ident) $body:block) => {
        $(#[$attribute])*
        #[test]
        fn $name() -> ::proptest::test_runner::TestCaseResult {
            let $path: $crate::hnsw::tests::property::test_runner_support::TestPath =
                concat!(module_path!(), "::", stringify!($name));
            $body
        }
    };
}

pub(crate) use forked_proptest;

/// What a caller asks for, before the run profile decides whether to fork.
///
/// This is `PropertyRunnerConfig` without its `fork`, which is the one field
/// the caller does not choose.
#[derive(Clone, Copy)]
struct PropertyRunnerRequest {
    test_path: TestPath,
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
}

fn run_test_with_profile<F>(request: PropertyRunnerRequest, test_runner: F) -> TestCaseResult
where
    F: FnOnce(Config, usize) -> TestCaseResult,
{
    let profile = property_run_profile(request.cases.get());
    let config = PropertyRunnerConfig {
        test_path: request.test_path,
        cases: request.cases,
        fork: profile.fork(),
        max_shrink_iters: request.max_shrink_iters,
        stack_size: request.stack_size,
    };
    run_test_with_config(config, test_runner)
}

fn run_test_with_profile_no_stack<F>(
    test_path: TestPath,
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    test_runner: F,
) -> TestCaseResult
where
    F: FnOnce(Config) -> TestCaseResult,
{
    let profile = property_run_profile(cases.get());
    test_runner(Config {
        cases: cases.get(),
        fork: profile.fork(),
        test_name: Some(test_path),
        max_shrink_iters: max_shrink_iters.get(),
        rng_seed: RngSeed::Fixed(PROPTEST_RNG_SEED),
        ..Config::default()
    })
}

/// Runs a mutation property test with custom configuration parameters.
///
/// `test_path` names the calling `#[test]`; see `TestPath`.
pub(crate) fn run_mutation_test(
    test_path: TestPath,
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
) -> TestCaseResult {
    run_test_with_profile(
        PropertyRunnerRequest {
            test_path,
            cases,
            max_shrink_iters,
            stack_size,
        },
        run_mutation_proptest_with_stack,
    )
}

/// Runs a search property test with custom configuration parameters.
///
/// `test_path` names the calling `#[test]`; see `TestPath`.
pub(crate) fn run_search_test(
    test_path: TestPath,
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
) -> TestCaseResult {
    run_test_with_profile_no_stack(test_path, cases, max_shrink_iters, run_search_proptest)
}

/// Runs an idempotency property test with custom configuration parameters.
///
/// `test_path` names the calling `#[test]`; see `TestPath`. This runner
/// never forks, so the name only shapes failure messages here, but it is
/// still supplied so that re-enabling `fork` cannot reintroduce #260.
pub(crate) fn run_idempotency_test(
    test_path: TestPath,
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
) -> TestCaseResult {
    let config = idempotency_runner_config(test_path, cases, max_shrink_iters, stack_size);
    run_test_with_config(config, run_idempotency_proptest_with_stack)
}

/// Runs a property test with the given configuration and strategy.
///
/// The failure message names the test from `config.test_name`. The fallback
/// only shapes that message: a genuinely missing name is caught by proptest
/// itself, which refuses to fork without one.
fn run_proptest<S, F>(config: Config, strategy: S, property: F) -> TestCaseResult
where
    S: proptest::strategy::Strategy,
    F: Fn(S::Value) -> TestCaseResult,
{
    let test_name = config.test_name.unwrap_or("unnamed hnsw proptest");
    let mut runner = TestRunner::new(config);
    runner
        .run(&strategy, property)
        .map_err(|err| map_test_error(err, test_name))
}

/// Maps `TestError` to `TestCaseError` with formatted messages.
fn map_test_error(err: TestError<impl std::fmt::Debug>, test_name: &str) -> TestCaseError {
    match err {
        TestError::Abort(reason) => TestCaseError::fail(format!("{test_name} aborted: {reason}")),
        TestError::Fail(reason, value) => TestCaseError::fail(format!(
            "{test_name} failed: {reason}; minimal input: {value:#?}"
        )),
    }
}

/// Spawns a property test runner on a dedicated thread with the given stack size.
///
/// Downcasts panic payloads to extract meaningful error messages.
fn spawn_with_stack<F>(name: &str, stack_size: usize, runner: F) -> TestCaseResult
where
    F: FnOnce() -> TestCaseResult + Send + 'static,
{
    let handle = std::thread::Builder::new()
        .name(name.into())
        .stack_size(stack_size)
        .spawn(runner)
        .map_err(|e| TestCaseError::fail(format!("failed to spawn {name} thread: {e}")))?;

    handle.join().map_err(|panic_payload| {
        // Try to downcast the panic payload to extract the actual panic message.
        let panic_msg = panic_payload.downcast_ref::<&str>().map_or_else(
            || {
                panic_payload
                    .downcast_ref::<String>()
                    .map_or_else(|| format!("{panic_payload:?}"), Clone::clone)
            },
            |message| (*message).to_owned(),
        );
        TestCaseError::fail(format!("{name} panicked: {panic_msg}"))
    })?
}

/// Runs a mutation property test with custom configuration and stack size.
fn run_mutation_proptest_with_stack(config: Config, stack_size: usize) -> TestCaseResult {
    spawn_with_stack("hnsw-mutation", stack_size, move || {
        run_mutation_proptest(config)
    })
}

fn run_mutation_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), mutation_plan_strategy()),
        |(fixture, plan)| run_mutation_property(&fixture, &plan),
    )
}

fn run_search_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), any::<u16>(), any::<u16>()),
        |(fixture, query_hint, k_hint)| {
            run_search_correctness_property(&fixture, query_hint, k_hint)
        },
    )
}

/// Configuration for property test runners that execute within dedicated threads.
///
/// Controls how property tests are executed, including test case count, forking behaviour,
/// shrinking limits, and thread stack size.
///
/// # Fields
/// - `test_path`: Libtest path of the calling `#[test]`; see `TestPath`
/// - `cases`: Number of test cases to execute in the property test run
/// - `fork`: Whether to run cases in a separate process (for crash isolation)
/// - `max_shrink_iters`: Maximum number of shrinking iterations when minimizing failures
/// - `stack_size`: Stack size in bytes for the dedicated property test runner thread
#[derive(Clone, Copy)]
struct PropertyRunnerConfig {
    test_path: TestPath,
    cases: TestCases,
    fork: bool,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
}

/// Builds idempotency runner configuration without per-case process forking.
///
/// Idempotency cases are always capped at `MAX_IDEMPOTENCY_CASES` in
/// `chutoro-core/src/hnsw/tests/property/test_runner_support/budget_selection.rs`
/// regardless of the `PROPTEST_CASES` environment variable. Proptest's
/// fork-based per-case process isolation therefore provides no benefit for
/// this test while multiplying process re-exec and the 96 MiB thread-respawn
/// cost from `spawn_with_stack` by the case count. That overhead caused
/// `hnsw_idempotency_preserved_proptest` to exceed the 600s nextest
/// slow-timeout override in the `property-tests-weekly` job when
/// `CHUTORO_PBT_FORK=true`.
///
/// # Example
///
/// ```rust,ignore
/// let config = idempotency_runner_config(
///     "hnsw::tests::property::tests::hnsw_idempotency_preserved_proptest",
///     TestCases::try_new(25000).expect("test cases must be > 0"),
///     ShrinkIterations::new(1024),
///     StackSize::try_new(96 * 1024 * 1024).expect("stack size must be >= minimum"),
/// );
///
/// assert!(!config.fork);
/// ```
fn idempotency_runner_config(
    test_path: TestPath,
    cases: TestCases,
    max_shrink_iters: ShrinkIterations,
    stack_size: StackSize,
) -> PropertyRunnerConfig {
    PropertyRunnerConfig {
        test_path,
        cases,
        fork: false,
        max_shrink_iters,
        stack_size,
    }
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
            test_name: Some(runner_config.test_path),
            max_shrink_iters: runner_config.max_shrink_iters.get(),
            rng_seed: RngSeed::Fixed(PROPTEST_RNG_SEED),
            ..Config::default()
        },
        runner_config.stack_size.get(),
    )
}

/// Runs an idempotency property test with custom configuration and stack size.
fn run_idempotency_proptest_with_stack(config: Config, stack_size: usize) -> TestCaseResult {
    spawn_with_stack("hnsw-idempotency", stack_size, move || {
        run_idempotency_proptest(config)
    })
}

fn run_idempotency_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), idempotency_plan_strategy()),
        |(fixture, plan)| run_idempotency_property(fixture, &plan),
    )
}

#[cfg(test)]
#[path = "runner_wrappers_tests.rs"]
mod tests;
