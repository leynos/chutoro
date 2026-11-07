"""Contract tests for the development route's host probe.

The development targets branch on the host operating system: on Linux they
select the fragment's mold linker and install mold. A probe that answers
nothing, or that answers and still reports failure, must stop the target rather
than let an empty or untrusted answer select the non-Linux branch. The standard
gates must stay unaffected by the probe, because they never use this route.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from dev_fast_routing_test import MAKEFILE_PATH, REPO_ROOT, _assert_unfragmented

DEVELOPMENT_TARGETS = ("dev-build", "dev-test", "install-dev-fast")


@pytest.mark.parametrize("target", DEVELOPMENT_TARGETS)
def test_development_targets_refuse_an_empty_host_probe(target: str) -> None:
    """An unanswerable host probe must stop the target, not pick a branch.

    ``DEV_FAST_HOST_OS`` is a ``?=`` default over the captured probe, and every
    use is a ``$(filter Linux,...)``, so an empty result reads as "not Linux".
    That would silently select the platform linker and skip the mold install on
    a Linux host whose probe failed. The check is per recipe because a
    file-wide ``$(error)`` would also fail the standard gates.
    """
    completed = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "--dry-run",
            "--always-make",
            "--file",
            str(MAKEFILE_PATH),
            "DEV_FAST_HOST_OS=",
            target,
            "CARGO=probe-cargo",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0, (
        f"{target} must fail when the host probe reports nothing: "
        f"{completed.stdout!r}"
    )
    assert "Cannot determine the development host OS" in completed.stderr, (
        f"{target} must report the failed probe: {completed.stderr!r}"
    )


#: A probe that answers ``uname -s`` but also reports failure. Its answer is
#: indistinguishable from a healthy Linux host once the exit status is dropped,
#: so a target that only checks for empty output would run the Linux branch
#: while the command that produced the answer had failed.
FAILING_PROBE = """#!/usr/bin/env bash
case "${1:-}" in
  -s) printf 'Linux\\n'; exit 1 ;;
  *) printf 'Linux\\n' ;;
esac
"""

#: A probe that prints nothing at all. The exit status is the axis under test,
#: so the caller supplies it: a silent success and a silent failure take
#: different paths through the recipe's empty and status checks.
SILENT_PROBE = """#!/usr/bin/env bash
case "${1:-}" in
  -s) exit __EXIT_STATUS__ ;;
  *) exit __EXIT_STATUS__ ;;
esac
"""


def _run_with_probe(
    target: str, probe: Path, *, overrides: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    """Dry-run one target with a controlled ``uname`` first on ``PATH``."""
    environment = os.environ.copy()
    environment["PATH"] = f"{probe.parent}{os.pathsep}{environment['PATH']}"
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            "--dry-run",
            "--always-make",
            "--file",
            str(MAKEFILE_PATH),
            *overrides,
            target,
            "CARGO=probe-cargo",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _failing_uname(tmp_path: Path) -> Path:
    """Write the probe above and return its directory's ``uname``."""
    probe_dir = tmp_path / "bin"
    probe_dir.mkdir()
    probe = probe_dir / "uname"
    probe.write_text(FAILING_PROBE, encoding="utf-8")
    probe.chmod(0o755)
    return probe


@pytest.mark.parametrize("target", DEVELOPMENT_TARGETS)
def test_development_targets_refuse_a_probe_that_exits_non_zero(
    tmp_path: Path, target: str
) -> None:
    """A probe that prints a value and still fails must stop the target.

    ``uname -s`` reporting failure while also emitting ``Linux`` is the
    inconsistent case: the text looks healthy, so only the recorded exit status
    reveals the problem. Dropping that status let ``dev-test`` select the Linux
    branch and its mold flag from a command that had failed.
    """
    completed = _run_with_probe(target, _failing_uname(tmp_path))

    assert completed.returncode != 0, (
        f"{target} must fail when the host probe exits non-zero: "
        f"{completed.stdout!r}"
    )
    assert "Cannot determine the development host OS" in completed.stderr, (
        f"{target} must report the failed probe: {completed.stderr!r}"
    )
    assert "exited" in completed.stderr, (
        f"{target} must name the exit status, not just emptiness: {completed.stderr!r}"
    )
    assert "probe-cargo" not in completed.stdout, (
        f"{target} must not run a command after a failed probe: {completed.stdout!r}"
    )


def _silent_uname(tmp_path: Path, exit_status: int) -> Path:
    """Write a ``uname`` that prints nothing and exits with ``exit_status``."""
    probe_dir = tmp_path / "bin"
    probe_dir.mkdir()
    probe = probe_dir / "uname"
    probe.write_text(
        SILENT_PROBE.replace("__EXIT_STATUS__", str(exit_status)), encoding="utf-8"
    )
    probe.chmod(0o755)
    return probe


@pytest.mark.parametrize("target", DEVELOPMENT_TARGETS)
@pytest.mark.parametrize("exit_status", (0, 1), ids=("silent-success", "silent-failure"))
def test_development_targets_refuse_a_silent_host_probe(
    tmp_path: Path, target: str, exit_status: int
) -> None:
    """A probe that prints nothing must stop the target at either exit status.

    ``DEV_FAST_HOST_OS`` is derived from this probe, so an empty answer reads as
    "not Linux" and silently selects the platform linker. The override tests
    above force that value directly and so never exercise this data flow; these
    cases let the probe produce it. A silent success and a silent failure are
    both covered, because the recipe checks emptiness and the recorded exit
    status separately and one check must not be able to mask the other.
    """
    completed = _run_with_probe(target, _silent_uname(tmp_path, exit_status))

    assert completed.returncode != 0, (
        f"{target} must fail when the host probe prints nothing "
        f"(exit {exit_status}): {completed.stdout!r}"
    )
    assert "Cannot determine the development host OS" in completed.stderr, (
        f"{target} must report the empty probe: {completed.stderr!r}"
    )
    assert "probe-cargo" not in completed.stdout, (
        f"{target} must not run a command after an empty probe: {completed.stdout!r}"
    )


def _counting_uname(directory: Path, count_file: Path) -> Path:
    """Write a ``uname`` that records one line per invocation."""
    probe_dir = directory / "bin"
    probe_dir.mkdir(parents=True)
    probe = probe_dir / "uname"
    probe.write_text(
        f"""#!/usr/bin/env bash
printf 'x\\n' >> {str(count_file)!r}
printf 'Linux\\n'
""",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    return probe


def _probe_invocations(target: str, probe: Path, count_file: Path) -> int:
    """Run one target and report how many times the probe was invoked."""
    if count_file.exists():
        count_file.unlink()
    completed = _run_with_probe(target, probe)
    assert completed.returncode == 0, (
        f"{target} failed with a working probe: {completed.stderr!r}"
    )
    return len(count_file.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("target", DEVELOPMENT_TARGETS)
def test_development_targets_add_no_probe_of_their_own(
    tmp_path: Path, target: str
) -> None:
    """Every reference in a recipe must read one captured answer.

    ``DEV_FAST_HOST_OS`` was a ``?=`` default over ``$(shell uname -s)``, which
    is recursively expanded, so each reference re-ran the command. That let one
    recipe check one answer and select its flags from another.

    The file-wide parse-time captures are shared by every target, so the
    baseline is a standard target's count rather than zero. What this pins is
    that evaluating a development recipe adds no further probe: both the check
    and the flag selection read the same captured value.
    """
    baseline_file = tmp_path / "baseline-calls"
    baseline = _probe_invocations(
        "build", _counting_uname(tmp_path / "baseline", baseline_file), baseline_file
    )

    count_file = tmp_path / "calls"
    count = _probe_invocations(
        target, _counting_uname(tmp_path / "target", count_file), count_file
    )

    assert count == baseline, (
        f"{target} invoked `uname` {count} times against a standard target's "
        f"{baseline}; a recipe reference re-ran the probe instead of reading "
        "the captured answer"
    )


def test_standard_targets_ignore_an_empty_host_probe() -> None:
    """The development route's precondition must not reach a merge gate.

    A parse-time check would fail every target in the file, including the
    gates that never use the fragment, so an unrelated probe failure could
    block a release. Standard targets evaluate the fragment's variables but
    must never validate them.
    """
    completed = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "--dry-run",
            "--always-make",
            "--file",
            str(MAKEFILE_PATH),
            "DEV_FAST_HOST_OS=",
            "build",
            "CARGO=probe-cargo",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        "a standard target must not fail on the development host probe: "
        f"{completed.stderr}"
    )
    _assert_unfragmented(completed.stdout, "build")


def test_the_probe_records_its_own_exit_status() -> None:
    """The status must come from the shell, not from ``.SHELLSTATUS``.

    ``.SHELLSTATUS`` only appeared in GNU Make 4.4. The earlier fallback
    treated "status unavailable" as "status acceptable", which made this exact
    case unsound on every older make: ``uname -s`` printing ``Linux`` and
    exiting non-zero selected the Linux branch from a command that had failed.
    Encoding the status in the probe's output has no version floor, so this
    pins the mechanism rather than a make-version-dependent behaviour that
    cannot be exercised on a modern test host.
    """
    source = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert "DEV_FAST_PROBE_STATUS=$$?" in source, (
        "the probe must append its own exit status to its output"
    )
    # A reintroduced `.SHELLSTATUS` read would restore the silent fallback on
    # any make older than 4.4, which is precisely the defect being guarded.
    assignments = [
        line
        for line in source.splitlines()
        if line.startswith("DEV_FAST_HOST_PROBE_STATUS") and "$(.SHELLSTATUS)" in line
    ]
    assert not assignments, (
        f"the probe status must not be read from .SHELLSTATUS: {assignments!r}"
    )


@pytest.mark.parametrize("target", DEVELOPMENT_TARGETS)
def test_development_targets_reject_a_failed_probe_without_shellstatus(
    tmp_path: Path, target: str
) -> None:
    """The status check must not depend on which make version is installed.

    ``SHELLSTATUS`` is a Make-internal variable, so it cannot be shadowed to
    reproduce an older make from here. What can be asserted is the contract
    that made the old path unsound: a probe that answers and fails is refused
    on this make, and the mechanism it relies on is one every make version
    supports.
    """
    completed = _run_with_probe(target, _failing_uname(tmp_path))
    source = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert completed.returncode != 0, (
        f"{target} must refuse a probe that answers and fails: {completed.stdout!r}"
    )
    assert "exited 1" in completed.stderr, (
        f"{target} must name the exit status: {completed.stderr!r}"
    )
    assert "DEV_FAST_HOST_STATUS_IS_KNOWN" not in source, (
        "the 'status unknown therefore acceptable' branch must not return"
    )
