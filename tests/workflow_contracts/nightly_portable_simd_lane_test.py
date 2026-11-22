"""Contract tests for the nightly portable-SIMD lane's gating and order.

`nightly-portable-simd` decides in its first step whether the lane has work,
then installs a toolchain and a linker before running four Cargo commands. A
step that loses the gate runs on a day the gate decided to skip, and an
installer that follows a Cargo step leaves that step without the linker the
build standard names. The lane also spent its whole history passing without
running its tests, so its step order is pinned rather than left to the next
edit.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections.abc as cabc
import typing as typ

import pytest
from rustup_install_reading import (
    commands_starting,
    runs_command,
    rustup_installs,
    shell_words,
)
from workflow_support import job, run_script, steps

#: The lane under contract.
WORKFLOW = "nightly-portable-simd.yml"
JOB = "nightly-portable-simd"

#: The condition that ties a step to the gate's decision.
GATE = "${{ steps.gate.outputs.should_run == 'true' }}"


def _required_index(
    job_steps: list[dict[str, typ.Any]],
    predicate: cabc.Callable[[dict[str, typ.Any]], bool],
    name: str,
) -> int:
    """Return the index of the only step satisfying the predicate.

    A second matching step is as much a defect as none: the contract would
    read the first and leave the duplicate's condition unchecked, so an
    ungated copy could run on a day the gate decided to skip.
    """
    matches = [
        index for index, step in enumerate(job_steps) if predicate(step)
    ]
    assert len(matches) == 1, (
        f"{WORKFLOW}:{JOB} must have exactly one {name} step, got "
        f"{matches!r} for {[run_script(step) for step in job_steps]!r}"
    )
    return matches[0]


def _condition(step: dict[str, typ.Any]) -> str | None:
    """Return a step's `if` condition, normalized for whitespace."""
    condition = step.get("if")
    return " ".join(condition.split()) if isinstance(condition, str) else None


def _is_gate(step: dict[str, typ.Any]) -> bool:
    """Report whether a step is the gate that decides whether to run."""
    return step.get("id") == "gate"


def _is_nightly_installer(step: dict[str, typ.Any]) -> bool:
    """Report whether a step installs the nightly toolchain.

    The shared reader joins continuations first, so a command split between
    `toolchain` and `install` is still recognised.
    """
    return bool(rustup_installs(run_script(step)))


def _is_mold_installer(step: dict[str, typ.Any]) -> bool:
    """Report whether a step installs the linker the Cargo steps name.

    The shared reader requires the path to start an executable command, so a
    step that only prints it does not count.
    """
    return runs_command(run_script(step), "scripts/install-mold.sh")


def _cargo_commands(step: dict[str, typ.Any]) -> list[list[str]]:
    """Return the decoded executable Cargo commands in one step."""
    return [
        shell_words(command)
        for command in commands_starting(run_script(step), "cargo")
    ]


def _has_cargo(
    step: dict[str, typ.Any],
    required: set[str],
    forbidden: frozenset[str] = frozenset(),
) -> bool:
    """Report whether an executable Cargo command has the required words."""
    for words in _cargo_commands(step):
        present = set(words)
        if required <= present and not forbidden & present:
            return True
    return False


#: The words a nightly Cargo test carries.
NIGHTLY_TEST = frozenset({"+nightly", "test"})

#: The dense provider selection.
DENSE_PROVIDER = frozenset({"-p", "chutoro-providers-dense"})

#: The compile-contract package selection.
COMPILE_CONTRACTS = frozenset({"-p", "chutoro-compile-contracts"})

#: The portable-SIMD gating test selection.
GATING_TEST = frozenset({"--test", "portable_simd_gating"})

#: The portable-SIMD feature selection.
PORTABLE_FEATURE = frozenset({"--features", "nightly_portable_simd"})


def _is_dense_test(step: dict[str, typ.Any]) -> bool:
    """Report whether a step tests the dense provider's portable SIMD backend."""
    return _has_cargo(
        step,
        NIGHTLY_TEST | DENSE_PROVIDER | PORTABLE_FEATURE,
        frozenset({"--no-run"}),
    )


def _is_contract_without_feature(step: dict[str, typ.Any]) -> bool:
    """Report whether a step tests the compile refusal before the feature."""
    return _has_cargo(
        step,
        NIGHTLY_TEST | COMPILE_CONTRACTS | GATING_TEST,
        frozenset({"--features"}),
    )


def _is_contract_with_feature(step: dict[str, typ.Any]) -> bool:
    """Report whether a step tests the compile contract with the feature."""
    return _has_cargo(
        step,
        NIGHTLY_TEST | COMPILE_CONTRACTS | GATING_TEST | PORTABLE_FEATURE,
    )


def _is_clippy(step: dict[str, typ.Any]) -> bool:
    """Report whether a step lints the dense provider under nightly."""
    return _has_cargo(
        step,
        frozenset({"+nightly", "clippy"}) | DENSE_PROVIDER | PORTABLE_FEATURE,
    )


#: Every gated step, named for its failure message and selected by predicate.
GATED_STEPS = (
    ("nightly installer", _is_nightly_installer),
    ("mold installer", _is_mold_installer),
    ("dense test", _is_dense_test),
    ("compile contract without feature", _is_contract_without_feature),
    ("compile contract with feature", _is_contract_with_feature),
    ("Clippy", _is_clippy),
)

#: The Cargo steps both installers must precede.
CARGO_STEPS = (
    "dense test",
    "compile contract without feature",
    "compile contract with feature",
    "Clippy",
)


def assert_lane_is_gated_and_ordered(job_definition: dict[str, typ.Any]) -> None:
    """Require the gate before every install and every Cargo step.

    Every selected step must carry the gate's condition. The two installers
    must also precede all four Cargo steps: an installer placed after one
    would leave that step without the tool it needs, even though the step is
    correctly gated.
    """
    job_steps = steps(job_definition)
    gate_at = _required_index(job_steps, _is_gate, "gate")
    positions = {
        name: _required_index(job_steps, predicate, name)
        for name, predicate in GATED_STEPS
    }

    conditions = {
        name: _condition(job_steps[index]) for name, index in positions.items()
    }
    assert set(conditions.values()) == {GATE}, (
        f"{WORKFLOW}:{JOB} must gate every tool installation and Cargo step "
        f"with {GATE!r}, got {conditions!r}"
    )

    installers = (positions["nightly installer"], positions["mold installer"])
    cargo_steps = [positions[name] for name in CARGO_STEPS]
    assert gate_at < min(installers), (
        f"{WORKFLOW}:{JOB} must decide before installing anything, got the "
        f"gate at {gate_at} and installers at {sorted(installers)}"
    )
    assert max(installers) < min(cargo_steps), (
        f"{WORKFLOW}:{JOB} must install both toolchains before every Cargo "
        f"step, got installers at {sorted(installers)} and Cargo steps at "
        f"{sorted(cargo_steps)}"
    )


def test_the_lane_gates_every_step_and_installs_before_using_it() -> None:
    """The real nightly job must carry the gate and the install order."""
    assert_lane_is_gated_and_ordered(job(WORKFLOW, JOB))


def test_a_split_installer_is_still_selected() -> None:
    """A continuation inside the command name must not hide the installer."""
    step = {"run": "rustup toolchain \\\n  install nightly --profile minimal"}

    assert _is_nightly_installer(step)


def test_a_printed_installer_is_not_an_installer() -> None:
    """A step that prints the script path must not count as installing mold."""
    step = {"run": "echo scripts/install-mold.sh"}

    assert not _is_mold_installer(step)


def test_an_assignment_prefixed_installer_is_selected() -> None:
    """An environment assignment before the command must not hide it."""
    step = {"run": "RUSTUP_TOOLCHAIN=stable rustup toolchain install nightly"}

    assert _is_nightly_installer(step)


def test_an_assignment_prefixed_mold_installer_is_selected() -> None:
    """The same prefix rule applies to the linker installer."""
    step = {"run": "MOLD_INSTALL_PREFIX=/tmp scripts/install-mold.sh"}

    assert _is_mold_installer(step)


def test_a_printed_cargo_test_is_not_a_test() -> None:
    """A step that prints a Cargo command runs no tests."""
    step = {
        "run": (
            "echo 'cargo +nightly test -p chutoro-providers-dense "
            "--features nightly_portable_simd'"
        )
    }

    assert not _is_dense_test(step)


def test_a_compile_only_dense_command_is_not_a_test() -> None:
    """`--no-run` compiles the tests without executing them."""
    step = {
        "run": (
            "cargo +nightly test -p chutoro-providers-dense "
            "--features nightly_portable_simd --no-run"
        )
    }

    assert not _is_dense_test(step)


def test_a_commented_cargo_lint_is_not_a_lint() -> None:
    """A commented-out Cargo lint runs nothing."""
    step = {
        "run": (
            "# cargo +nightly clippy -p chutoro-providers-dense "
            "--features nightly_portable_simd"
        )
    }

    assert not _is_clippy(step)


def test_the_gate_records_its_decision_in_the_job_summary() -> None:
    """A skipped green run must say so rather than leave no evidence.

    The lane reports success in seconds when it does nothing, so the gate's
    own line in the job summary is the only place the skip is visible. Three
    appends are required: forced dispatch, a recent `main` commit, and no
    recent commit.
    """
    job_steps = steps(job(WORKFLOW, JOB))
    gate_at = _required_index(job_steps, _is_gate, "gate")
    gate_script = run_script(job_steps[gate_at])
    assert gate_script.count("GITHUB_STEP_SUMMARY") == 3, (
        f"{WORKFLOW}:{JOB} must record each gate decision in the job "
        f"summary, got {gate_script!r}"
    )


def _copied_steps() -> list[dict[str, typ.Any]]:
    """Return a shallow copy of the real job's steps, safe to mutate."""
    return [dict(step) for step in steps(job(WORKFLOW, JOB))]


def test_an_ungated_installer_is_rejected() -> None:
    """Removing one installer's condition must fail the contract."""
    job_steps = _copied_steps()
    mold_at = _required_index(job_steps, _is_mold_installer, "mold installer")
    job_steps[mold_at].pop("if")

    with pytest.raises(AssertionError, match="must gate every"):
        assert_lane_is_gated_and_ordered({"steps": job_steps})


def test_an_installer_after_a_cargo_step_is_rejected() -> None:
    """A correctly gated installer must still precede every Cargo step."""
    job_steps = _copied_steps()
    mold_at = _required_index(job_steps, _is_mold_installer, "mold installer")
    job_steps.append(job_steps.pop(mold_at))

    with pytest.raises(AssertionError, match="must install both toolchains"):
        assert_lane_is_gated_and_ordered({"steps": job_steps})


def test_a_duplicate_step_is_rejected() -> None:
    """A second matching step must fail even when the first is gated."""
    job_steps = _copied_steps()
    dense_at = _required_index(job_steps, _is_dense_test, "dense test")
    duplicate = dict(job_steps[dense_at])
    duplicate.pop("if")
    job_steps.append(duplicate)

    with pytest.raises(AssertionError, match="exactly one dense test"):
        assert_lane_is_gated_and_ordered({"steps": job_steps})