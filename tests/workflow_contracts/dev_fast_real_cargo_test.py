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

from dev_fast_make_support import DEV_FAST_CONFIG, REPO_ROOT, _pinned_toolchain


def _run_cargo(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Cargo in the repository with the caller's arguments."""
    return subprocess.run(
        ["cargo", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _installed_toolchains() -> set[str] | None:
    """Return the toolchain names rustup reports, or ``None`` if it failed.

    ``rustup toolchain list`` prints one name per line, usually with a host
    triple appended and optionally a ``(default)`` or ``(active)`` marker.
    Splitting each line on whitespace keeps the name and drops the marker, so
    callers can compare names rather than substrings.
    """
    listed = subprocess.run(
        ["rustup", "toolchain", "list"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        return None
    return {
        fields[0] for fields in map(str.split, listed.stdout.splitlines()) if fields
    }


def _is_toolchain_installed(toolchain: str, installed: set[str]) -> bool:
    """Report whether a pin is installed as itself or with a host triple.

    A substring test would also match a longer name that merely contains the
    pin, so a host holding only ``nightly-2026-05-28-something`` would skip
    the real Cargo check while reporting the pin as present. Accepting the
    bare name and the name plus a ``-``-separated suffix covers rustup's
    actual output without accepting unrelated releases.

    >>> _is_toolchain_installed("nightly-1", {"nightly-1-x86_64-host"})
    True
    >>> _is_toolchain_installed("nightly-1", {"nightly-10-x86_64-host"})
    False
    """
    return any(
        name == toolchain or name.startswith(f"{toolchain}-") for name in installed
    )


def test_the_pinned_toolchain_resolves_the_fragment_in_real_cargo() -> None:
    """The fragment must select Cranelift for real Cargo, not only in make.

    A fragment that kept its ``--config`` argument but lost
    ``codegen-backend`` would satisfy every dry-run assertion while building
    with the default backend, silently removing the speed-up the route exists
    to provide. Asking the pinned nightly to resolve the profile is the
    cheapest execution that can tell the difference.
    """
    toolchain = _pinned_toolchain()
    installed = _installed_toolchains()
    if installed is None or not _is_toolchain_installed(toolchain, installed):
        pytest.skip(f"the pinned development toolchain {toolchain} is not installed")

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
