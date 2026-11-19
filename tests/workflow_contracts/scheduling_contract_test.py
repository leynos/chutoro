"""Contract tests for the HNSW scheduling-regression CI gate.

The two Rust regressions in ``hnsw::cpu::trim_tests`` are the runtime evidence
that trim scoring stays inside the insertion critical section. Evidence that
runs nowhere protects nothing: a renamed test, a removed Make target, or a CI
step deleted beside it would all leave the suite green.

These tests read the Makefile recipe and the ``build-test`` job in
``.github/workflows/ci.yml``. They pin the target, one invocation per fully
qualified exact-match test name, the zero-test failure flag on each, the
unconditional execution of the step, and the dependency on an installed
``cargo-nextest``. They also require default features so the gate cannot become
collateral damage when the coverage action's feature set changes.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from workflow_support import job, run_script, steps

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"

#: The target that runs both scheduling regressions.
TARGET = "test-scheduling-contract"

#: The exact test names the filter must select.
REGRESSIONS = (
    "insertion_trim_scoring_runs_on_the_insert_mutex_holder",
    "direct_trim_scoring_stays_on_the_calling_worker",
)

#: The fully qualified nextest names, which the recipes must match exactly.
QUALIFIED_REGRESSIONS = {
    "insertion_trim_scoring_runs_on_the_insert_mutex_holder": (
        "hnsw::cpu::trim_tests::insertion_path::"
        "insertion_trim_scoring_runs_on_the_insert_mutex_holder"
    ),
    "direct_trim_scoring_stays_on_the_calling_worker": (
        "hnsw::cpu::trim_tests::direct_trim_scoring_stays_on_the_calling_worker"
    ),
}


def makefile_text() -> str:
    """Return the Makefile source this contract reads.

    Reading once keeps every assertion below looking at the same text, so a
    failure names the file once rather than reporting a diagnostic from a
    different read.
    """
    return MAKEFILE.read_text(encoding="utf-8")


def target_recipe(text: str, target: str) -> str:
    """Return the tab-indented recipe of one Make target.

    The target line itself may carry a ``##`` help comment. Recipe lines are
    the tab-indented lines that follow, backslash continuations included; a
    missing target or an empty recipe fails rather than returning no recipe,
    because an empty string would satisfy every substring check vacuously.

    Examples
    --------
    >>> target_recipe("build:\\n\\tcargo build\\n", "build")
    'cargo build'
    >>> target_recipe("build:\\n", "build")
    Traceback (most recent call last):
    ...
    ValueError: build has no recipe
    """
    pattern = re.compile(
        rf"^{re.escape(target)}:[^\n]*\n(?P<recipe>(?:[ \t]+[^\n]*\n)*)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"{target} is not declared with a recipe")
    lines = [line.strip() for line in match.group("recipe").splitlines()]
    recipe = " ".join(line for line in lines if line)
    if not recipe:
        raise ValueError(f"{target} has no recipe")
    return recipe


def phony_targets(text: str) -> set[str]:
    """Return the names a ``.PHONY`` declaration covers.

    GNU make collapses continued declarations, so the continuation backslashes
    are removed before splitting. A target absent from this set can be skipped
    when a file of the same name exists.
    """
    declared = re.findall(r"^\.PHONY:((?:.*\\\n)*.*)$", text, re.MULTILINE)
    return set(" ".join(declared).replace("\\\n", " ").split())


def test_the_target_retires_when_the_filter_selects_nothing() -> None:
    """A filter that matches nothing must fail rather than pass.

    ``cargo nextest`` exits successfully when a filter selects no test. That
    default would let a rename retire the evidence silently, so the flag is
    part of the contract rather than a local convenience. Each invocation
    carries its own flag; a shared one would cover only the last command.
    """
    recipe = target_recipe(makefile_text(), TARGET)
    assert recipe.count("--no-tests=fail") == len(REGRESSIONS), (
        "every scheduling invocation must fail when it selects no test"
    )


def test_each_regression_runs_in_its_own_invocation() -> None:
    """One combined filter cannot vouch for a single missing name.

    A disjunction still selects the surviving test when the other name stops
    matching, so ``--no-tests=fail`` never fires. A separate invocation makes
    each name, and the test it must select, fail on its own.
    """
    recipe = target_recipe(makefile_text(), TARGET)
    assert recipe.count("nextest run") == len(REGRESSIONS), (
        "the target must run one nextest invocation per regression"
    )
    for name in REGRESSIONS:
        assert recipe.count(f"test(={QUALIFIED_REGRESSIONS[name]})") == 1, (
            f"{name} must be selected by exactly one exact-match invocation"
        )


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
    assert "nextest run" in recipe
    assert "-p chutoro-core" in recipe
    assert "--lib" in recipe
    assert "--all-features" not in recipe


def test_the_target_is_phony() -> None:
    """A file named like the target must not satisfy it."""
    assert TARGET in phony_targets(makefile_text())


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
    """Fail unless `step` exists and runs without a step-level condition.

    These steps have no side effect worth guarding: they either run the
    evidence or the job has not done its work. A condition, including
    ``if: false``, would let the step disappear while the contract still
    found its command in the file.
    """
    assert step is not None, f"build-test must run {command}"
    condition = step.get("if")
    assert condition is None, (
        f"{command} must run unconditionally; it is guarded by {condition!r}"
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
def test_each_regression_is_a_real_test_in_the_scanned_rust_sources(name: str) -> None:
    """The selected names must still exist as tests in the Rust source.

    This is a second, independent reading: the Makefile could name a test that
    was renamed in the Rust source, and the zero-test flag would only catch
    that at CI time. Reading the declaration here fails at review time.
    """
    sources = (ROOT / "chutoro-core" / "src" / "hnsw" / "cpu" / "trim_tests").glob(
        "*.rs"
    )
    declarations = {path.read_text(encoding="utf-8") for path in sources}
    assert any(
        re.search(rf"fn\s+{re.escape(name)}\s*\(", source)
        for source in declarations
    ), f"{name} must still be declared under cpu/trim_tests"