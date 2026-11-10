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

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from dev_fast_make_support import _skip_unless_pin_installed
from dev_fast_probe_support import (
    CRANELIFT_ARGUMENT,
    assert_probe_inputs_are_real,
    linker_missing_reason,
    nextest_missing_reason,
    probe_environment,
    run_development_target,
    write_probe_repository,
)


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
    assert CRANELIFT_ARGUMENT in completed.stderr, (
        "dev-build must hand the fragment's Cranelift backend to rustc; got:\n"
        f"{completed.stderr}"
    )
    binary = directory / "target" / "debug" / "devprobe"
    assert binary.is_file(), f"dev-build produced no binary at {binary}"
    ran = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert ran.stdout.strip() == "devprobe ok", (
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
    """
    _skip_unless_dev_test_can_run()
    directory = write_probe_repository(tmp_path / "probe")

    completed = run_development_target(
        directory,
        probe_environment(directory, tmp_path / "home"),
        "dev-test",
        build_jobs="--cargo-verbose",
    )

    assert completed.returncode == 0, (
        f"dev-test failed on a minimal crate:\n{completed.stdout}\n{completed.stderr}"
    )
    assert CRANELIFT_ARGUMENT in completed.stderr, (
        "dev-test must compile the probe's test with the fragment's Cranelift "
        f"backend, not only run it; got:\n{completed.stderr}"
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
        ["cargo", "build", "--bin", "devprobe", "--verbose"],
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
