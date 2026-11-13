"""Contracts that run the development targets end to end.

``dev_fast_routing_test`` reads ``make --dry-run`` output, and
``dev_fast_real_cargo_test`` only asks Cargo to resolve the fragment. Neither
proves the two targets that exist for a developer's edit-compile-test loop
actually compile and test a crate. A recipe that named the fragment correctly
but lost its ``RUSTFLAGS`` composition, or whose nextest invocation never
reached the pinned toolchain, would satisfy both.

The contracts here run the real recipes against a tiny probe crate, which
``dev_fast_probe_support`` builds. They are the slowest in the suite and skip
rather than provision: ``make install-dev-fast`` is the route that installs the
pinned nightly, its Cranelift component and the mold linker, so a host missing
any of them must not reach the network from a test. CI is such a host - it
installs the same nightly pin for Whitaker, without the Cranelift component.

The probe crate proves a recipe executes. It cannot prove what the production
``DEV_TEST_FILTER`` selects, because the filter names a case in this
repository's own source and the probe carries none of it. That is the one gap
the selection contract below closes, and it is the gap that matters most: a
``test(=...)`` clause that misspells the poisoned-lock regression still
balances, still reads as an exclusion, and excludes nothing, so ``make
dev-test`` then reaches the case that aborts the process.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from dev_fast_listing_support import _parse_nextest_listing
from dev_fast_make_support import (
    MAKEFILE_PATH,
    _make_output,
    _nextest_filter_expression,
    _skip_unless_pin_installed,
)
from dev_fast_probe_support import (
    CRANELIFT_ARGUMENT,
    PROBE_CRATE,
    PROBE_GREETING,
    assert_probe_inputs_are_real,
    crate_compilations,
    harness_compilations,
    linker_missing_reason,
    nextest_missing_reason,
    probe_environment,
    run_development_target,
    write_probe_repository,
)
from dev_fast_selection_contract import assert_development_selection_is_standard_minus_one

#: The development filter, shaped like the production one but without the
#: ``package(chutoro-core)`` scope. The probe crate is one synthetic package,
#: and nextest rejects a ``package(...)`` that names nothing in the tree, so the
#: production filter cannot be used there. The shape is what the recipe's own
#: plumbing needs: a benchmark clause and one exclusion group.
PROBE_FILTER = "not kind(bench) & not (test(=the_route_runs_tests))"


def _skip_unless_dev_test_can_run() -> None:
    """Skip the calling case when ``dev-test``'s prerequisites are absent."""
    reason = nextest_missing_reason()
    if reason is not None:
        pytest.skip(reason)
    _skip_unless_linker_available("dev-test links with the pinned nightly")
    _skip_unless_pin_installed(
        "dev-test runs the suite on the pinned nightly", needs_cranelift=True
    )


def _skip_unless_linker_available(reason: str) -> None:
    """Skip the calling case when this host has no linker to build with."""
    missing = linker_missing_reason()
    if missing is not None:
        pytest.skip(f"{missing}; {reason}")


def test_dev_build_compiles_and_links_the_probe_with_cranelift(tmp_path: Path) -> None:
    """``dev-build`` must produce a running binary from the Cranelift backend.

    A recipe that dropped ``--config`` would still compile, because the probe
    builds under the default backend too; only the compiler arguments and a
    binary that runs separate the two. Both are asserted, so a route that
    reached Cranelift but failed to link - mold's failure mode - cannot pass.
    """
    _skip_unless_linker_available("dev-build links with the pinned nightly")
    _skip_unless_pin_installed(
        "dev-build compiles with the pinned nightly", needs_cranelift=True
    )
    directory = write_probe_repository(tmp_path / "probe")

    completed = run_development_target(
        directory, probe_environment(directory, tmp_path / "home"), "dev-build"
    )

    assert completed.returncode == 0, (
        f"dev-build failed on a minimal crate:\n{completed.stdout}\n{completed.stderr}"
    )
    compiled = crate_compilations(completed.stderr, PROBE_CRATE)
    assert compiled, (
        "dev-build must print the rustc invocation it ran for the probe; got:\n"
        f"{completed.stderr}"
    )
    assert all(CRANELIFT_ARGUMENT in line for line in compiled), (
        "every probe compilation must carry the fragment's Cranelift backend; "
        f"got {compiled!r}"
    )
    binary = directory / "target" / "debug" / PROBE_CRATE
    assert binary.is_file(), f"dev-build produced no binary at {binary}"
    ran = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert ran.stdout.strip() == PROBE_GREETING, (
        f"the development binary must run: {ran.returncode} {ran.stdout!r}"
    )


def test_dev_test_runs_the_suite_through_the_pinned_toolchain(
    tmp_path: Path,
) -> None:
    """``dev-test`` must reach nextest and run the probe's tests.

    The recipe hands the fragment to Cargo once for itself and again for
    nextest's inner invocation, because nextest starts a second Cargo that
    does not inherit the outer configuration. A recipe that dropped the inner
    ``--config`` would still build with the outer one and then test whatever
    nextest's own defaults selected, which is the defect this asserts against.
    The compiler arguments are checked too, because nextest reporting a pass
    says nothing about which backend compiled the test.

    ``RecipeOverrides`` supplies the probe-shaped filter: the production one
    scopes its exclusion with ``package(chutoro-core)``, and nextest rejects a
    ``package(...)`` naming nothing in the tree, so ``dev-test`` would fail
    before the recipe's own plumbing was exercised.
    """
    _skip_unless_dev_test_can_run()
    directory = write_probe_repository(tmp_path / "probe")

    completed = run_development_target(
        directory,
        probe_environment(directory, tmp_path / "home"),
        "dev-test",
        {"BUILD_JOBS": "--cargo-verbose", "DEV_TEST_FILTER": PROBE_FILTER},
    )

    assert completed.returncode == 0, (
        f"dev-test failed on a minimal crate:\n{completed.stdout}\n{completed.stderr}"
    )
    harnesses = harness_compilations(completed.stderr, PROBE_CRATE)
    assert harnesses, (
        "dev-test must print the rustc invocation that built the probe's test "
        f"harness; got:\n{completed.stderr}"
    )
    assert all(CRANELIFT_ARGUMENT in line for line in harnesses), (
        "the probe's test harness must be compiled with the fragment's "
        f"Cranelift backend, not only run; got {harnesses!r}"
    )
    assert "the_route_runs_tests" in completed.stderr, (
        f"dev-test must report the probe's test by name; got:\n{completed.stderr}"
    )
    assert "1 test run: 1 passed" in completed.stderr, (
        f"dev-test must run and pass the probe's suite; got:\n{completed.stderr}"
    )


def test_a_missing_backend_argument_is_detectable_in_the_probe_output(
    tmp_path: Path,
) -> None:
    """The Cranelift assertion must fail for a route that never selects it.

    Without this control the two cases above would also pass against a Cargo
    that mentioned ``codegen-backend`` for an unrelated reason, so the
    assertion is exercised against a plain stable build that does not.
    """
    directory = write_probe_repository(tmp_path / "probe")
    environment = probe_environment(directory, tmp_path / "home")

    built = subprocess.run(
        ["cargo", "build", "--bin", PROBE_CRATE, "--verbose"],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert built.returncode == 0, built.stderr
    assert CRANELIFT_ARGUMENT not in built.stderr, (
        "a plain stable build must not name the development backend, or the "
        f"end-to-end assertion proves nothing:\n{built.stderr}"
    )


def test_the_probe_inputs_match_the_files_the_recipes_read() -> None:
    """The probe must copy inputs the repository actually ships and reads."""
    assert_probe_inputs_are_real()



# --- Selection of the repository's own suite -------------------------------

REPO_ROOT = MAKEFILE_PATH.parent
#: The filter the standard gate runs, and therefore the baseline the
#: development route is measured against.
STANDARD_FILTER = "not kind(bench)"
#: ``cargo nextest list`` builds every test binary before it can ask them what
#: they contain, so this is a compile budget rather than a parsing one. It is
#: generous because the two selections run back to back on one workspace and
#: the second reuses the first's artefacts; a cold cache pays it twice.
LIST_TIMEOUT_SECONDS = 900

requires_repository_selection = pytest.mark.skipif(
    not os.environ.get("CHUTORO_DEV_FAST_E2E"),
    reason="opt-in: set CHUTORO_DEV_FAST_E2E=1 to enumerate the real suite",
)


def _build_and_list_nextest_selection(*, filter_expression: str) -> set[str]:
    """Compile the workspace, then return the tests nextest selects for it.

    The compilation is the point, not a hidden cost. ``cargo nextest list``
    documents that it builds every test binary before it can query it for its
    contents, and it has no listing-without-building mode, so a contract that
    compares two selections pays for a build twice over one workspace. The
    name says so: the side effect happens here, and ``_parse_nextest_listing``
    stays a read.

    The returned set is the selection ``nextest run`` would execute, including
    the binary prefix, which is what makes membership direct evidence rather
    than an inference from counts. Counts move whenever an unrelated test is
    added, so they cannot show which case a filter dropped.

    ``-T oneline`` is asked for rather than left to ``auto``, which reads the
    terminal: a captured pipe already yields one line per test, but saying so
    keeps the parse from depending on how the process was launched.
    """
    completed = subprocess.run(
        [
            "cargo",
            "nextest",
            "list",
            "--all-targets",
            "--all-features",
            "--color",
            "never",
            "-T",
            "oneline",
            "-E",
            filter_expression,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=LIST_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"nextest list rejected {filter_expression!r}: {completed.stderr}"
    )
    return _parse_nextest_listing(completed.stdout)


@pytest.fixture(scope="module")
def dev_test_filter() -> str:
    """Return the filter expression ``make dev-test`` really passes to nextest.

    The tokenizing helpers are shared with the routing suite rather than
    re-derived here: both suites read the same Makefile, and a second parser
    could disagree with the first on a recipe that gained another ``-E``.
    """
    output = _make_output("dev-test", overrides=("DEV_FAST_HOST_OS=Linux",))
    return _nextest_filter_expression(output)


@requires_repository_selection
def test_development_selection_is_the_standard_selection_minus_one_case(
    dev_test_filter: str,
) -> None:
    """The development route drops exactly the poisoned-lock regression.

    This is the assertion the filter text cannot make. A rename that orphans
    the exclusion leaves the expression balanced, still excluding *nothing*
    that exists, and ``make dev-test`` then runs the case that aborts the
    Cranelift process. Comparing live selections to live names catches it.

    Both selections are computed on the stable pin, which is deliberate: test
    names come from the source, so they do not depend on the codegen backend,
    and enumerating under the pinned nightly would pay a Cranelift build for
    evidence the probe contracts above already buy. What this covers is the
    filter's membership; what they cover is the route executing.
    """
    dev_selection = _build_and_list_nextest_selection(
        filter_expression=dev_test_filter
    )
    standard_selection = _build_and_list_nextest_selection(
        filter_expression=STANDARD_FILTER
    )

    assert_development_selection_is_standard_minus_one(
        dev_selection, standard_selection
    )
