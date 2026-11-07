"""End-to-end contracts for the real development targets.

Every other development-route contract reads a ``make --dry-run`` transcript.
That proves what the Makefile *says* and nothing about what Cargo, nextest, or
the Cranelift backend *do*. This module runs the actual recipes on a host that
has the pinned nightly and its codegen backend installed, and it is opt-in
through ``CHUTORO_DEV_FAST_E2E`` because a cold Cranelift build costs far more
than a contract suite should spend by default.

The selection assertions are the load-bearing part. ``make dev-test`` exists to
run the accelerated suite minus the one case Cranelift cannot unwind, and
``make test`` exists to keep running that case. Reading the filter text cannot
distinguish "excluded" from "misspelt": both leave the case unrun. Only nextest's
own selection, computed from live test names, separates them.

Run via ``make test-workflow-contracts`` with ``CHUTORO_DEV_FAST_E2E=1``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_FAST_CONFIG = "tools/dev-fast/config.toml"
POISONED_LOCK_TEST = (
    "hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring"
)
SMOKE_TEST = "hnsw::graph::test_helpers::tests::delete_node_returns_ok_false_for_missing_node"
E2E_ENABLE_ENV = "CHUTORO_DEV_FAST_E2E"
APP_BINARY = REPO_ROOT / "target" / "debug" / "chutoro-cli"
# The compiler records its own identity in the linked binary's ``.comment``
# section. Asserting it is what separates "the recipe ran" from "the recipe ran
# on the pinned Cranelift nightly": a build that silently fell back to stable
# would still produce a binary, and still exit zero.
CRANELIFT_BACKEND_MARKER = "with cranelift"

requires_dev_fast_e2e = pytest.mark.skipif(
    not os.environ.get(E2E_ENABLE_ENV),
    reason=f"opt-in: set {E2E_ENABLE_ENV}=1 to run the real development targets",
)


def _run_make(
    target: str,
    *,
    overrides: tuple[str, ...] = (),
    timeout: int,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one Make target, or evaluate its recipe without running it.

    ``DEV_FAST_HOST_OS`` is passed explicitly so the route is exercised on a
    host whose ``uname`` probe is unavailable, which is the same override the
    dry-run contracts use. It does not change which commands run.
    """
    command = ["make", "--no-print-directory", "--file", "Makefile"]
    if dry_run:
        command.extend(("--dry-run", "--always-make"))
    command.extend(overrides)
    command.append(target)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _nextest_selection(*, filter_expression: str, timeout: int) -> set[str]:
    """Return the fully qualified test names nextest selects, verbatim.

    ``cargo nextest list`` prints the same selection ``nextest run`` executes,
    including the binary prefix, so membership in this set is direct evidence
    rather than an inference from counts. Colours are disabled because the
    escape codes would otherwise split the package prefix from the test path
    and make an exact suffix match silently fail.
    """
    command = [
        "cargo",
        "nextest",
        "list",
        "--all-targets",
        "--all-features",
        "--color",
        "never",
        "-E",
        filter_expression,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        f"nextest list rejected {filter_expression!r}: {completed.stderr}"
    )
    return {
        stripped
        for line in completed.stdout.splitlines()
        if "::" in (stripped := ANSI_ESCAPE.sub("", line))
    }


def _selected(selection: set[str], test_name: str) -> set[str]:
    """Return the selection entries naming ``test_name``."""
    return {entry for entry in selection if entry.endswith(test_name)}


@pytest.fixture(scope="module")
def dev_test_filter() -> str:
    """Return the filter expression ``make dev-test`` really passes to nextest.

    This is a dry run, so it is cheap enough to keep outside the opt-in gate.
    The expression it reads is the same one the real recipe builds, and the
    end-to-end test below proves nextest accepts it.
    """
    completed = _run_make(
        "dev-test",
        overrides=("DEV_FAST_HOST_OS=Linux",),
        timeout=120,
        dry_run=True,
    )
    assert completed.returncode == 0, completed.stderr
    marker = "-E '"
    start = completed.stdout.find(marker)
    assert start != -1, f"dev-test must pass -E: {completed.stdout}"
    start += len(marker)
    end = completed.stdout.find("'", start)
    assert end != -1, f"dev-test filter is unterminated: {completed.stdout}"
    return completed.stdout[start:end]


@requires_dev_fast_e2e
def test_development_selection_is_the_standard_selection_minus_one_case(
    dev_test_filter: str,
) -> None:
    """The development route drops exactly the poisoned-lock regression.

    This is the assertion the filter text cannot make. A rename that orphans
    ``POISONED_LOCK_TEST`` leaves the expression balanced, still excluding
    *nothing* that exists, and ``make dev-test`` then runs the case that aborts
    the Cranelift process. Comparing live selections to live names catches it.

    Both selections are computed on the stable pin, which is deliberate: test
    names come from the source, so they do not depend on the codegen backend,
    and enumerating under the pinned nightly would pay a Cranelift build for
    evidence the smoke run below already buys. What this contract covers is the
    filter's membership; what the smoke run covers is the real route executing.
    """
    dev_selection = _nextest_selection(filter_expression=dev_test_filter, timeout=900)
    standard_selection = _nextest_selection(
        filter_expression="not kind(bench)", timeout=900
    )

    assert _selected(dev_selection, POISONED_LOCK_TEST) == set(), (
        "dev-test must not select the poisoned-lock regression"
    )
    assert len(_selected(standard_selection, POISONED_LOCK_TEST)) == 1, (
        "the standard gate must select the poisoned-lock regression"
    )
    assert dev_selection == standard_selection - _selected(
        standard_selection, POISONED_LOCK_TEST
    ), (
        "dev-test must select every standard case except the poisoned-lock "
        f"regression; unexpected difference: "
        f"{dev_selection ^ standard_selection}"
    )


@requires_dev_fast_e2e
def test_development_test_route_builds_and_runs_under_the_pinned_backend(
    dev_test_filter: str,
) -> None:
    """``make dev-test`` accepts the filter and runs one real case to green.

    The whole suite is bounded rather than unbounded: a cold Cranelift build of
    every target is not what this contract is for, and a full run has its own
    ceiling. Keeping the production filter as the base and adding one
    restriction is what makes this a smoke run of the real route rather than a
    different route.
    """
    smoke_expression = f"({dev_test_filter}) & test(={SMOKE_TEST})"
    completed = _run_make(
        "dev-test",
        overrides=(
            "DEV_FAST_HOST_OS=Linux",
            f"DEV_TEST_FILTER={smoke_expression}",
        ),
        timeout=3600,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, (
        f"dev-test smoke run failed with {completed.returncode}:\n{output[-4000:]}"
    )
    assert "error: expected close parenthesis" not in output, (
        f"nextest rejected the development filter:\n{output[-4000:]}"
    )
    assert "failed to initiate panic" not in output, (
        f"the Cranelift route reached a case it cannot unwind:\n{output[-4000:]}"
    )
    assert SMOKE_TEST.rsplit("::", 1)[-1] in output, (
        f"the smoke run did not reach the requested case:\n{output[-4000:]}"
    )
    assert "1 passed" in output or "PASS" in output, (
        f"the smoke run did not pass:\n{output[-4000:]}"
    )


def _compiler_comment(binary: Path) -> str:
    """Return the ``.comment`` section of ``binary``, or empty when unreadable.

    ``readelf`` is part of binutils and is not guaranteed on every host, so the
    caller treats an unreadable section as a skip rather than a failure. The
    section is where the compiler leaves its version string, which is the only
    durable proof of which backend produced the binary.
    """
    completed = subprocess.run(
        ["readelf", "-p", ".comment", str(binary)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout


@requires_dev_fast_e2e
def test_development_build_produces_a_binary_from_the_pinned_backend() -> None:
    """``make dev-build`` really compiles the binary with Cranelift.

    This is the half of the requested end-to-end coverage that the dry-run
    contracts cannot supply. They prove the recipe passes ``--config
    tools/dev-fast/config.toml``; they cannot prove Cargo honoured it, and a
    fragment that silently stopped selecting the backend would leave every
    text-level assertion passing while the build reverted to stable Rust.

    The binary is removed first so the compiler identity in its ``.comment``
    section is evidence about *this* invocation, not a leftover from an earlier
    one.
    """
    APP_BINARY.unlink(missing_ok=True)

    completed = _run_make(
        "dev-build",
        overrides=("DEV_FAST_HOST_OS=Linux",),
        timeout=3600,
    )
    assert completed.returncode == 0, (
        f"dev-build failed with {completed.returncode}:\n"
        f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
    )
    assert APP_BINARY.is_file(), (
        f"dev-build must produce {APP_BINARY.relative_to(REPO_ROOT)}"
    )

    comment = _compiler_comment(APP_BINARY)
    if not comment:
        pytest.skip("readelf is unavailable, so the backend identity is unreadable")
    assert CRANELIFT_BACKEND_MARKER in comment, (
        "dev-build must compile with the Cranelift backend; the binary reports "
        f"these compilers instead:\n{comment}"
    )
