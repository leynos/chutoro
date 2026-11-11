"""Contracts that hand the development fragment to a real Cargo.

``dev_fast_routing_test`` reads `make --dry-run` output, which proves the
command line carries ``--config tools/dev-fast/config.toml`` but never that
Cargo acts on it. The contracts here execute Cargo itself, so a fragment that
lost its ``[unstable]`` opt-in, its ``codegen-backend`` key, or its toolchain
pin fails with evidence from the tool rather than from make's echo.

Provisioning the pinned nightly is what ``make install-dev-fast`` is for, so a
host that has not provisioned it skips rather than reaching the network.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import subprocess

import pytest

from dev_fast_make_support import (
    DEV_FAST_CONFIG,
    REPO_ROOT,
    _pinned_toolchain,
    _skip_unless_pin_installed,
)


def _run_cargo(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Cargo in the repository with the caller's arguments."""
    return subprocess.run(
        ["cargo", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_the_pinned_toolchain_resolves_the_fragment_in_real_cargo() -> None:
    """The fragment must select Cranelift for real Cargo, not only in make.

    A fragment that kept its ``--config`` argument but lost
    ``codegen-backend`` would satisfy every dry-run assertion while building
    with the default backend, silently removing the speed-up the route exists
    to provide. Asking the pinned nightly to resolve the profile is the
    cheapest execution that can tell the difference.
    """
    _skip_unless_pin_installed("Cargo cannot resolve the fragment without it")
    toolchain = _pinned_toolchain()

    completed = _run_cargo(
        f"+{toolchain}",
        "-Z",
        "unstable-options",
        "--config",
        DEV_FAST_CONFIG,
        "config",
        "get",
        "profile.dev",
    )

    assert completed.returncode == 0, (
        f"Cargo could not resolve the development profile: {completed.stderr}"
    )
    assert 'profile.dev.codegen-backend = "cranelift"' in completed.stdout, (
        "real Cargo must read the Cranelift backend from the fragment; got "
        f"{completed.stdout!r}"
    )
