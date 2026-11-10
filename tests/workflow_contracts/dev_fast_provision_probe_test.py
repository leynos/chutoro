"""Contracts for the skip probes the execution-level suites guard themselves with.

The end-to-end and linker suites compile a real crate, so a host that lacks
the pinned nightly must skip rather than reach the network. Skipping too
eagerly would hide the route; skipping too late would fail on a host that
never claimed to be provisioned. CI showed the second failure: it installs the
same nightly pin for the Whitaker suite, so a toolchain-only probe passes, and
the build then dies on the missing Cranelift codegen component.

These contracts drive the probes against a stand-in ``rustup`` so every
provisioning state is stated without needing a real toolchain, and without
asking the host's own rustup anything. The stand-in reports what the scenario
describes, including failure, so a probe that cannot distinguish an absent
component from an unreadable listing is visible here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dev_fast_make_support import (
    REPO_ROOT,
    ComponentProbeError,
    _has_cranelift_backend,
    _pinned_toolchain,
)

#: A ``rustup`` that answers ``toolchain list`` and ``component list`` from a
#: scenario described by environment variables, so one script covers every
#: provisioning state. ``STUB_COMPONENT_STATUS`` lets a scenario fail the
#: component listing while the toolchain itself still looks installed.
RUSTUP_STAND_IN = """#!/usr/bin/env bash
case "${1:-}" in
  toolchain)
    printf '%b\\n' "${STUB_TOOLCHAINS:-}"
    ;;
  component)
    printf '%b\\n' "${STUB_COMPONENTS:-}"
    exit "${STUB_COMPONENT_STATUS:-0}"
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
UNPROVISIONED_COMPONENTS = (
    "rustfmt-x86_64-unknown-linux-gnu\\nclippy-x86_64-unknown-linux-gnu"
)

#: The source the child interpreter runs, so the probe is exercised in a real
#: subprocess with the stand-in first on its ``PATH`` rather than by editing
#: the parent's environment.
PROBE_CALL = (
    "from dev_fast_make_support import _has_cranelift_backend; "
    "from dev_fast_make_support import _pinned_toolchain; "
    "print(_has_cranelift_backend(_pinned_toolchain()))"
)


def _with_stand_in_rustup(
    tmp_path: Path, *, components: str, component_status: int = 0
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
        "STUB_COMPONENT_STATUS": str(component_status),
    }
    return subprocess.run(
        [
            # The probe's module imports pytest at import time, so the child
            # must run on the interpreter that is already running these
            # tests. A bare "python3" finds the system interpreter instead,
            # which has no site-packages of its own.
            sys.executable,
            "-c",
            PROBE_CALL,
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


def test_a_failed_component_listing_raises_rather_than_reporting_absence(
    tmp_path: Path,
) -> None:
    """An unreadable listing must be distinguishable from an absent component.

    A host can claim the toolchain is installed while rustup cannot answer for
    its components, and the two facts call for different responses: absence
    means the developer has not provisioned the route, while a failed listing
    means the probe or rustup is broken and the caller must say so.
    """
    completed = _with_stand_in_rustup(
        tmp_path, components=UNPROVISIONED_COMPONENTS, component_status=1
    )

    assert completed.returncode != 0, (
        "a failed listing must not answer for the component; got "
        f"{completed.stdout!r}"
    )
    assert "ComponentProbeError" in completed.stderr, (
        "the failure must name the probe error at the test boundary; got "
        f"{completed.stderr!r}"
    )


def test_an_unreadable_listing_is_reported_as_a_probe_error(tmp_path: Path) -> None:
    """The raised type must carry the toolchain and the reason it could not read."""
    completed = _with_stand_in_rustup(
        tmp_path, components=UNPROVISIONED_COMPONENTS, component_status=1
    )

    assert completed.returncode != 0, completed.stdout
    assert _pinned_toolchain() in completed.stderr, (
        "the error must name the toolchain whose components could not be read; "
        f"got {completed.stderr!r}"
    )


def test_a_caller_handling_skips_does_not_swallow_the_probe_error() -> None:
    """Handling a skip must not also handle an unreadable probe.

    Callers of the guard catch pytest's skip signal to decide the case cannot
    run. If the probe error shared that type, a broken listing would be
    silently reported as an unprovisioned host and the defect would ship.
    """
    try:
        raise ComponentProbeError("rustup could not list components")
    except pytest.skip.Exception:  # pragma: no cover - the assert must not run
        raise AssertionError("a probe error must not be caught as a skip") from None
    except ComponentProbeError:
        handled = "probe error"
    assert handled == "probe error"
