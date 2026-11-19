"""Contract tests for the HNSW scheduling-regression CI gate.

The two Rust regressions in ``hnsw::cpu::trim_tests`` are the runtime evidence
that trim scoring stays inside the insertion critical section. Evidence that
runs nowhere protects nothing: a renamed test, a removed Make target, or a CI
step deleted beside it would all leave the suite green.

These tests read the Makefile recipe and the ``build-test`` job in
``.github/workflows/ci.yml``. They pin the target, one invocation per fully
qualified exact-match test name, the zero-test failure flag on each, the
unconditional and blocking execution of the step within an unconditional job,
and the dependency on an installed ``cargo-nextest``. They also require default
features so the gate cannot become collateral damage when the coverage action's
feature set changes.

The reading itself, with the cases that keep its filtering honest, lives in
``scheduling_reading``.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from scheduling_reading import (
    QUALIFIED_REGRESSIONS,
    REGRESSIONS,
    TARGET,
    declares_test,
    invocations_missing_flag,
    is_nextest_invocation,
    makefile_text,
    nextest_invocations,
    phony_targets,
    recipe_commands,
    selected_regression,
    target_recipe,
)
from workflow_support import ROOT, job, run_script, steps


def test_the_target_retires_when_the_filter_selects_nothing() -> None:
    """A filter that matches nothing must fail rather than pass.

    ``cargo nextest`` exits successfully when a filter selects no test. That
    default would let a rename retire the evidence silently, so the flag is
    part of the contract rather than a local convenience. Each invocation is
    checked on its own: counting flags across the whole recipe would accept a
    command that carries the flag twice beside one that omits it.
    """
    invocations = nextest_invocations()
    assert len(invocations) == len(REGRESSIONS), (
        "the target must run one nextest invocation per regression"
    )
    without_flag = invocations_missing_flag(invocations)
    assert not without_flag, (
        f"scheduling invocation(s) {without_flag} must pass --no-tests=fail"
    )


def test_a_flag_on_another_invocation_does_not_cover_a_missing_one() -> None:
    """The flag is a per-invocation requirement, not a recipe allowance.

    One command carrying the flag twice must not cover a second command that
    omits it; that is exactly the shape a whole-recipe count accepts.
    """
    invocations = [
        "cargo nextest run -E 'test(=a)' --no-tests=fail --no-tests=fail",
        "cargo nextest run -E 'test(=b)'",
    ]
    assert invocations_missing_flag(invocations) == [1], (
        "only the unflagged invocation may be reported"
    )


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(
            "echo $(CARGO) nextest run -p chutoro-core --lib",
            id="a-mentioned-command",
        ),
        pytest.param(
            "-$(CARGO) nextest run -p chutoro-core --lib",
            id="an-ignored-error",
        ),
        pytest.param(
            "./scripts/nextest-runner nextest run",
            id="an-unrelated-runner",
        ),
    ],
)
def test_only_a_command_that_runs_nextest_counts(command: str) -> None:
    """A mention is not an invocation, and Make's `-` hides a failure.

    Counting the text anywhere in a recipe accepts ``echo $(CARGO) nextest
    run ...``, which runs nothing while every argument assertion passes.
    Accepting a leading ``-`` is the same problem with a worse outcome: the
    regression runs, fails, and the target still succeeds.
    """
    assert not is_nextest_invocation(command), (
        f"{command!r} must not count as a scheduling invocation"
    )


def test_the_recipe_does_not_ignore_errors() -> None:
    """Make suppresses a failure for any recipe line beginning with `-`.

    The CI step and job checks reject ``continue-on-error``, but neither sees
    a recipe-level prefix, so the target's own commands are read for one.
    """
    commands = recipe_commands(makefile_text(), TARGET)
    ignored = [command for command in commands if command.startswith("-")]
    assert not ignored, (
        f"the scheduling recipe must not ignore errors: {ignored}"
    )


def test_each_regression_runs_in_its_own_invocation() -> None:
    """One combined filter cannot vouch for a single missing name.

    A disjunction still selects the surviving test when the other name stops
    matching, so ``--no-tests=fail`` never fires. Each invocation must select
    exactly one regression with an expression naming nothing else, and the
    invocations must cover every regression once: a per-name total spread
    across commands would accept two selectors in one command and an
    unrelated test in the other.
    """
    invocations = nextest_invocations()
    assert len(invocations) == len(REGRESSIONS), (
        "the target must run one nextest invocation per regression"
    )
    chosen = [selected_regression(command) for command in invocations]
    assert sorted(chosen) == sorted(REGRESSIONS), (
        "the invocations must cover each regression exactly once; "
        f"got {chosen}"
    )


def test_a_widened_expression_is_rejected() -> None:
    """A disjunction with all() keeps an invocation non-empty forever.

    Both invocations could carry ``test(=name) | all()`` and pass every count:
    the expression selects every test, so the zero-test flag never fires once
    the named regression is renamed. The whole expression has to name the
    regression, not merely contain its selector.
    """
    widened = (
        "cargo nextest run -E "
        f"'test(={QUALIFIED_REGRESSIONS[REGRESSIONS[0]]}) | all()'"
    )
    with pytest.raises(ValueError, match="must select exactly one regression"):
        selected_regression(widened)


def test_both_scheduling_regressions_are_selected() -> None:
    """Both test names must appear in the selection expressions.

    Dropping either name would leave the other's single observation standing
    in for a property that needs both a direct and a real-insertion witness.
    The substring form is rejected as well: nextest treats it as a partial
    match, so a renamed regression could leave a like-named fixture passing.
    """
    recipe = target_recipe(makefile_text(), TARGET)
    missing = [name for name in REGRESSIONS if name not in recipe]
    assert not missing, f"the filter must select {missing}"
    for name in REGRESSIONS:
        assert f"test({name})" not in recipe, (
            f"{name} must be matched exactly, not as a substring"
        )


def test_the_target_runs_the_core_library_with_default_features() -> None:
    """The gate is scoped to the core library and does not widen features.

    The regressions are feature-independent. ``--all-features`` would tie this
    evidence to the coverage action's configuration, so a feature change could
    alter what the scheduling gate builds.
    """
    recipe = target_recipe(makefile_text(), TARGET)
    assert "nextest run" in recipe, "the target must run nextest"
    assert "-p chutoro-core" in recipe, "the gate must select chutoro-core"
    assert "--lib" in recipe, "the gate must run the library target"
    assert "--all-features" not in recipe, (
        "the gate must not widen the feature set"
    )


def test_the_target_is_phony() -> None:
    """A file named like the target must not satisfy it."""
    assert TARGET in phony_targets(makefile_text()), (
        f"{TARGET} must be declared phony"
    )


def test_the_target_is_declared_once() -> None:
    """A later duplicate recipe would override the one this contract reads.

    GNU make keeps the last recipe declared for a repeated target and only
    warns. The contract reads the first declaration, so a duplicate could
    replace the gated recipe with a command the checks never saw.
    """
    declarations = [
        line
        for line in makefile_text().splitlines()
        if line.startswith(f"{TARGET}:")
    ]
    assert len(declarations) == 1, (
        f"{TARGET} must be declared once; found {len(declarations)} recipes"
    )


def _build_test_steps() -> list[dict[str, object]]:
    """Return the ``build-test`` steps that run a script, in order.

    The step mapping is kept rather than only its command because a step can
    carry a condition that skips it. Reading the command alone would accept a
    step guarded by ``if: false`` as evidence that the gate runs.
    """
    return [
        step
        for step in steps(job("ci.yml", "build-test"))
        if run_script(step).strip()
    ]


def _build_test_runs() -> list[str]:
    """Return the run scripts of the ``build-test`` job, in order."""
    return [run_script(step).strip() for step in _build_test_steps()]


def _step_running(command: str) -> dict[str, object] | None:
    """Return the ``build-test`` step whose script is exactly `command`."""
    return next(
        (
            step
            for step in _build_test_steps()
            if run_script(step).strip() == command
        ),
        None,
    )


def _assert_unconditional(step: dict[str, object] | None, command: str) -> None:
    """Fail unless `step` exists and its failure would fail the job.

    These steps have no side effect worth guarding: they either run the
    evidence or the job has not done its work. A condition, including
    ``if: false``, would let the step disappear while the contract still
    found its command in the file, and ``continue-on-error`` would let a
    failing regression leave the job green.
    """
    assert step is not None, f"build-test must run {command}"
    condition = step.get("if")
    assert condition is None, (
        f"{command} must run unconditionally; it is guarded by {condition!r}"
    )
    assert "continue-on-error" not in step, (
        f"{command} must fail the job when it fails; it is non-blocking"
    )


def test_ci_runs_the_scheduling_target_beside_the_workflow_contracts() -> None:
    """The gate must run in the build job, not only in a local recipe."""
    runs = _build_test_runs()
    assert f"make {TARGET}" in runs, (
        f"build-test must run make {TARGET}; it runs {runs}"
    )
    assert "make test-workflow-contracts" in runs, (
        "the scheduling gate belongs with the workflow contracts"
    )
    _assert_unconditional(_step_running(f"make {TARGET}"), f"make {TARGET}")


def test_the_build_test_job_is_not_conditional() -> None:
    """A job-level condition skips every step the contract relies on.

    The step check reads only the scheduling step's own guard. A condition
    on ``build-test`` itself would skip the whole job, including that step,
    and the evidence would never run.
    """
    definition = job("ci.yml", "build-test")
    condition = definition.get("if")
    assert condition is None, (
        f"build-test must run unconditionally; it is guarded by {condition!r}"
    )
    assert "continue-on-error" not in definition, (
        "build-test must fail when a gate fails; the job tolerates failure"
    )


def test_ci_installs_nextest_before_the_scheduling_target() -> None:
    """The target cannot select tests if the nextest binary is absent.

    ``scripts/install-nextest.sh`` is the job's only nextest installation. A
    scheduling step placed before it would fail for a reason unrelated to the
    regressions it exists to run, and either step being skipped would break
    the chain just as surely as running them in the wrong order.
    """
    runs = _build_test_runs()
    install = next(
        (index for index, run in enumerate(runs) if "install-nextest.sh" in run),
        None,
    )
    target = next(
        (index for index, run in enumerate(runs) if run == f"make {TARGET}"),
        None,
    )
    assert install is not None, "build-test must install cargo-nextest"
    assert target is not None, f"build-test must run make {TARGET}"
    assert install < target, (
        "cargo-nextest must be installed before the scheduling target runs"
    )
    installer = next(
        (
            step
            for step in _build_test_steps()
            if "install-nextest.sh" in run_script(step)
        ),
        None,
    )
    _assert_unconditional(installer, "scripts/install-nextest.sh")


@pytest.mark.parametrize("name", REGRESSIONS)
def test_each_regression_is_a_real_test_in_the_scanned_rust_sources(
    name: str,
) -> None:
    """The selected names must still exist as tests in the Rust source.

    This is a second, independent reading: the Makefile could name a test that
    was renamed in the Rust source, and the zero-test flag would only catch
    that at CI time. Reading the declaration here fails at review time. The
    declaration must carry a test attribute, because a plain function of the
    same name is never registered with the harness.
    """
    sources = (
        ROOT / "chutoro-core" / "src" / "hnsw" / "cpu" / "trim_tests"
    ).glob("*.rs")
    declarations = [path.read_text(encoding="utf-8") for path in sources]
    assert any(declares_test(source, name) for source in declarations), (
        f"{name} must still be declared as a test under cpu/trim_tests"
    )


def test_a_plain_function_does_not_satisfy_the_declaration_check() -> None:
    """A matching name without an attribute is not a registered test.

    ``--no-tests=fail`` cannot fire for a function the harness never
    discovered, so the declaration check would pass while the Makefile filter
    selected nothing.
    """
    source = f"fn {REGRESSIONS[0]}() {{}}"
    assert not declares_test(source, REGRESSIONS[0]), (
        "an unattributed function must not read as a declared test"
    )