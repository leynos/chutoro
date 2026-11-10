"""Contracts for the skip probes the execution-level suites guard themselves with.

The end-to-end and linker suites compile a real crate, so a host that lacks
the pinned nightly must skip rather than reach the network. Skipping too
eagerly would hide the route; skipping too late would fail on a host that
never claimed to be provisioned. CI showed the second failure: it installs the
same nightly pin for the Whitaker suite, so a toolchain-only probe passes, and
the build then dies on the missing Cranelift codegen component.

These contracts drive the probes against a stand-in ``rustup`` so both sides
of the boundary are stated without needing a real toolchain.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dev_fast_make_support import REPO_ROOT, _has_cranelift_backend, _pinned_toolchain

#: A ``rustup`` that answers ``toolchain list`` and ``component list`` from a
#: scenario described by two environment variables, so one script covers every
#: provisioning state without a real toolchain.
RUSTUP_STAND_IN = """#!/usr/bin/env bash
case "${1:-}" in
  toolchain)
    printf '%b\\n' "${STUB_TOOLCHAINS:-}"
    ;;
  component)
    printf '%b\\n' "${STUB_COMPONENTS:-}"
    ;;
  *)
    exit 2
    ;;
esac
"""

#: What a provisioned host reports: the pin plus its Cranelift component,
#: which rustup names with a host triple appended.
PROVISIONED_COMPONENTS = "rustc-codegen-cranelift-x86_64-unknown-linux-gnu"

#: What CI reports: the same pin, installed for Whitaker with stable
#: components only.
UNPROVISIONED_COMPONENTS = "rustfmt-x86_64-unknown-linux-gnu\\nclippy-x86_64-unknown-linux-gnu"


def _with_stand_in_rustup(
    tmp_path: Path, *, components: str
) -> subprocess.CompletedProcess[str]:
    """Run the Cranelift probe against a stand-in whose toolchain is present."""
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir(parents=True, exist_ok=True)
    rustup = binary_directory / "rustup"
    rustup.write_text(RUSTUP_STAND_IN, encoding="utf-8")
    rustup.chmod(0o755)
    toolchain = _pinned_toolchain()
    environment = {
        "PATH": f"{binary_directory}:/usr/bin:/bin",
        # The probe lives in this suite's own support module, which the child
        # interpreter finds only if the suite directory is on its path.
        "PYTHONPATH": str(Path(__file__).resolve().parent),
        "STUB_TOOLCHAINS": f"{toolchain}-x86_64-unknown-linux-gnu (default)",
        "STUB_COMPONENTS": components,
    }
    return subprocess.run(
        [
            # The probe's module imports pytest at import time, so the child
            # must run on the interpreter that is already running these
            # tests. A bare "python3" finds the system interpreter instead,
            # which has no site-packages of its own.
            sys.executable,
            "-c",
            "from dev_fast_make_support import _has_cranelift_backend; "
            "from dev_fast_make_support import _pinned_toolchain; "
            "print(_has_cranelift_backend(_pinned_toolchain()))",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_the_cranelift_probe_accepts_a_provisioned_toolchain(tmp_path: Path) -> None:
    """A toolchain holding the component must be reported as usable.

    Without this side, a probe that always answered ``False`` would skip every
    execution-level suite on a developer host and still look correct.
    """
    completed = _with_stand_in_rustup(tmp_path, components=PROVISIONED_COMPONENTS)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True", (
        "a toolchain carrying rustc-codegen-cranelift must not be skipped; "
        f"got {completed.stdout!r}"
    )


def test_the_cranelift_probe_rejects_the_component_set_ci_installs(
    tmp_path: Path,
) -> None:
    """The pin present without the component must not be reported as usable.

    This is the CI state: the channel is installed for the Whitaker suite, so
    a toolchain-only probe passes and the build then fails with "failed to
    find a ``codegen-backends`` folder". The probe has to see the difference.
    """
    completed = _with_stand_in_rustup(tmp_path, components=UNPROVISIONED_COMPONENTS)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", (
        "the same pin without the Cranelift component must not be treated as "
        f"provisioned; got {completed.stdout!r}"
    )


def test_the_cranelift_probe_is_not_fooled_by_a_longer_component_name(
    tmp_path: Path,
) -> None:
    """Matching is per component name, not a substring of the listing.

    Rustup appends a host triple after a ``-``, so a suffix is expected. A
    name that merely begins with the probe's name and continues without that
    separator is a different component, and reporting it as the backend would
    turn a skip into a build failure.
    """
    completed = _with_stand_in_rustup(
        tmp_path,
        components="rustc-codegen-craneliftx-x86_64-unknown-linux-gnu",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", (
        "a differently named component must not be read as the Cranelift "
        f"backend; got {completed.stdout!r}"
    )


def test_the_component_probe_fails_closed_when_rustup_cannot_list() -> None:
    """An unreadable component listing must not claim the backend exists.

    Reporting a missing toolchain as provisioned turns a skip into a build
    failure on exactly the hosts the guard exists to protect.
    """
    assert not _has_cranelift_backend("a-toolchain-that-is-not-installed"), (
        "an unknown toolchain must not be reported as carrying the backend"
    )


def test_an_unknown_toolchain_is_not_reported_as_installed() -> None:
    """The probe must not invent a component for a channel rustup rejects."""
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            ["rustup", "component", "list", "--installed", "--toolchain", "nope"],
            check=True,
            capture_output=True,
            text=True,
        )
