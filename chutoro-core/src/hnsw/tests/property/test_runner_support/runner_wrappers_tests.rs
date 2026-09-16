//! Unit tests for the proptest runner wrappers.
//!
//! These live beside `runner_wrappers.rs` rather than inside it because the
//! repository caps a source file at 400 lines and `self_named_module_files`
//! rules out a `runner_wrappers/` directory next to `runner_wrappers.rs`.
//! `#[path]` keeps it a child module, so it still sees private items.

use super::*;
use proptest::test_runner::FileFailurePersistence;

const SAMPLE_TEST_PATH: TestPath = "hnsw::tests::property::sample_proptest";

#[test]
fn idempotency_runner_config_disables_forking() {
    let cases = TestCases::try_new(25000).expect("test cases must be > 0");
    let max_shrink_iters = ShrinkIterations::new(1024);
    let stack_size = StackSize::try_new(96 * 1024 * 1024).expect("stack size must be >= minimum");

    let config = idempotency_runner_config(SAMPLE_TEST_PATH, cases, max_shrink_iters, stack_size);

    assert_eq!(config.cases, cases);
    assert!(!config.fork);
    assert_eq!(config.max_shrink_iters, max_shrink_iters);
    assert_eq!(config.stack_size, stack_size);
}

/// The stacked runner hands proptest the caller's test path.
///
/// Without it a forked run aborts before its first case, so this asserts
/// the field rather than the absence of a panic: dropping `test_name`
/// from the built config fails here and nowhere else (#260).
#[test]
fn the_stacked_runner_names_its_test() {
    let config = PropertyRunnerConfig {
        test_path: SAMPLE_TEST_PATH,
        cases: TestCases::try_new(4).expect("test cases must be > 0"),
        fork: true,
        max_shrink_iters: ShrinkIterations::new(8),
        stack_size: StackSize::try_new(96 * 1024 * 1024).expect("stack size must be >= minimum"),
    };

    let mut captured = None;
    let outcome = run_test_with_config(config, |proptest_config, _stack_size| {
        captured = Some(proptest_config);
        Ok(())
    });

    assert!(outcome.is_ok(), "the capturing runner cannot fail");
    let built = captured.expect("the runner is called exactly once");
    assert_eq!(built.test_name, Some(SAMPLE_TEST_PATH));
    assert!(built.fork);
}

/// The stackless runner hands proptest the caller's test path.
///
/// `run_search_test` is the only user of this path and it forks under
/// the weekly profile, so the same omission would silence it (#260).
#[test]
fn the_stackless_runner_names_its_test() {
    let mut captured = None;
    let outcome = run_test_with_profile_no_stack(
        SAMPLE_TEST_PATH,
        TestCases::try_new(4).expect("test cases must be > 0"),
        ShrinkIterations::new(8),
        |proptest_config| {
            captured = Some(proptest_config);
            Ok(())
        },
    );

    assert!(outcome.is_ok(), "the capturing runner cannot fail");
    let built = captured.expect("the runner is called exactly once");
    assert_eq!(built.test_name, Some(SAMPLE_TEST_PATH));
}

/// The stacked runner sizes its reject budget from its case count.
///
/// proptest's default is a flat 1024 for the whole run however many cases are
/// asked for. The HNSW mutation suite exhausted it at 7,326 successes the
/// first time it ever drew a case, so dropping this field would leave the
/// weekly lane aborting rather than finishing (#260).
///
/// The assertion is against the rule rather than a literal, so it holds
/// whatever `PROPTEST_CASES` the surrounding run happens to set.
#[test]
fn the_stacked_runner_scales_its_reject_budget() {
    let config = PropertyRunnerConfig {
        test_path: SAMPLE_TEST_PATH,
        cases: TestCases::try_new(25_000).expect("test cases must be > 0"),
        fork: true,
        max_shrink_iters: ShrinkIterations::new(8),
        stack_size: StackSize::try_new(96 * 1024 * 1024).expect("stack size must be >= minimum"),
    };

    let mut captured = None;
    let outcome = run_test_with_config(config, |proptest_config, _stack_size| {
        captured = Some(proptest_config);
        Ok(())
    });

    assert!(outcome.is_ok(), "the capturing runner cannot fail");
    let built = captured.expect("the runner is called exactly once");
    assert_eq!(
        built.max_global_rejects,
        max_global_rejects_for(built.cases),
        "a deep run left on proptest's flat default aborts before it finishes"
    );
}

/// The stackless runner sizes its reject budget from its case count.
///
/// `run_search_test` is the only user of this path, and it reached 736
/// successes before the flat default stopped it (#260).
#[test]
fn the_stackless_runner_scales_its_reject_budget() {
    let mut captured = None;
    let outcome = run_test_with_profile_no_stack(
        SAMPLE_TEST_PATH,
        TestCases::try_new(25_000).expect("test cases must be > 0"),
        ShrinkIterations::new(8),
        |proptest_config| {
            captured = Some(proptest_config);
            Ok(())
        },
    );

    assert!(outcome.is_ok(), "the capturing runner cannot fail");
    let built = captured.expect("the runner is called exactly once");
    assert_eq!(
        built.max_global_rejects,
        max_global_rejects_for(built.cases),
        "a deep run left on proptest's flat default aborts before it finishes"
    );
}

forked_proptest! {
    /// A forked run executes at least one case, in a child process.
    ///
    /// This is the end-to-end half of #260, and it discriminates on the two
    /// things that can go wrong rather than on the absence of a panic.
    ///
    /// That a case ran is the difference between `Fail` and `Abort`. proptest
    /// returns `Abort` when the child appended nothing to its replay file,
    /// which is what selecting a test that does not exist produces; a missing
    /// name does not get that far, because proptest refuses to fork without one.
    ///
    /// That the case ran in a child is the difference between the two `Fail`
    /// reasons. A child's failure comes back through the replay file, which
    /// records that the case failed and not what it said, so the parent reports
    /// its own wording. Seeing this property's own message instead would mean
    /// the run never left this process and `fork` was not in effect.
    ///
    /// It is defined with `forked_proptest!` on purpose. That is the macro the
    /// HNSW suites take their paths from, and it is the only test here that
    /// forks, so the derivation and the mechanism stand or fall together.
    fn a_forked_run_executes_at_least_one_case(test_path) {
        const CASE_MARKER: &str = "this case ran in the parent process";

        let config = Config {
            cases: 1,
            fork: true,
            test_name: Some(test_path),
            max_shrink_iters: 64,
            // Keep the deliberate failure out of the repository's
            // proptest-regressions files.
            failure_persistence: Some(Box::new(FileFailurePersistence::Off)),
            rng_seed: RngSeed::Fixed(PROPTEST_RNG_SEED),
            ..Config::default()
        };

        let outcome = TestRunner::new(config)
            .run(&any::<u8>(), |_| Err(TestCaseError::fail(CASE_MARKER)));

        // Reported as an error rather than a panic, so the failure arrives
        // as the runner's own result type and says which of the two it was.
        let Err(error) = outcome else {
            return Err(TestCaseError::fail(
                "the property fails on every input, so the run cannot succeed",
            ));
        };

        match error {
            TestError::Fail(reason, _) if reason.message() == CASE_MARKER => Err(
                TestCaseError::fail("the case ran in this process, so `fork` was not in effect"),
            ),
            TestError::Fail(_, _) => Ok(()),
            TestError::Abort(reason) => Err(TestCaseError::fail(format!(
                "no case ran in the forked child: {reason}"
            ))),
        }
    }
}
