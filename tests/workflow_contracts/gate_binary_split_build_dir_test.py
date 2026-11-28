"""Run the gate-binary suites with Cargo's build directory split from its target.

Cargo 1.91 stabilised ``build.build-dir``, which moves intermediate artefacts
out of the target directory and leaves the finished binaries behind. The test
harnesses and other hashed artefacts live in ``<build-dir>/debug/deps``, while
``<target-dir>/debug`` holds the binaries a caller runs. An integration test
must therefore take a sibling binary from Cargo's compile-time
``CARGO_BIN_EXE_<name>`` value. Reconstructing the path from ``current_exe()``
and scanning ``deps`` instead can select the gate binary's own
``<name>-<hash>`` unit-test harness, which exits successfully after printing
``running 0 tests``.

This contract runs both gate suites under that layout. It asserts that the
suites actually executed, that no harness ran in place of a gate, and that the
finished binaries and hashed artefacts landed in their separate directories.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``CARGO_BUILD_BUILD_DIR`` is stable from this Cargo version. Earlier
#: releases reject the variable, so a contract run there would test a layout
#: the tool cannot produce.
MINIMUM_CARGO = (1, 91)

#: Each suite under test, mapped to the gate binary it exercises. The binary
#: name is both the ``CARGO_BIN_EXE_<name>`` value Cargo embeds and the file
#: name it writes into the target directory's debug profile directory.
#:
#: Only the benchmark gate remains. The nightly Kani gate this contract was
#: written alongside was deleted when the nightly workflow became
#: unconditional, and `kani_suite_cli` replaces nothing here: it takes its
#: binary from `CARGO_BIN_EXE_kani_suite` too, but its own suite already
#: asserts that path is an executable file, so listing it would repeat a check
#: rather than cover a second gate.
GATE_SUITES: dict[str, str] = {
    "benchmark_regression_gate_cli": "benchmark_regression_gate",
}

#: Cargo appends a 16-hex-digit metadata hash to every artefact compiled for
#: the ``deps`` directory, optionally followed by ``.d`` for a dep-info file or
#: the platform executable suffix. This is the harness shape a filename scan
#: could mistake for a binary.
HASHED_ARTEFACT = re.compile(r"-[0-9a-f]{16}(?:\.d|\.exe)?$")

#: Variables a caller may have set that would make this contract repeat a
#: layout instead of creating two. ``GITHUB_OUTPUT`` is cleared because the
#: gates write step outputs to it when it is present.
CLEARED_VARIABLES = (
    "CARGO_TARGET_DIR",
    "CARGO_BUILD_BUILD_DIR",
    "CARGO_BUILD_TARGET",
    "GITHUB_OUTPUT",
)


def _executable(stem: str) -> str:
    """Return the platform's file name for a compiled binary."""
    return f"{stem}.exe" if os.name == "nt" else stem


class CargoVersionProbeError(RuntimeError):
    """``cargo --version`` ran but could not report a usable version."""


def _cargo_version() -> tuple[int, int]:
    """Return Cargo's major and minor version.

    Raises:
        CargoVersionProbeError: when the command fails or its output carries
            no version. A present but broken Cargo is a probe failure, not an
            unprovisioned host, so it must not be reported as a skip.
    """
    completed = subprocess.run(
        ["cargo", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CargoVersionProbeError(
            "cargo --version failed: "
            f"exit {completed.returncode}: {completed.stderr.strip()}"
        )
    match = re.search(r"\bcargo (\d+)\.(\d+)", completed.stdout)
    if match is None:
        raise CargoVersionProbeError(
            "cargo --version printed no usable version: "
            f"{completed.stdout.strip()!r}"
        )
    return int(match.group(1)), int(match.group(2))


def skip_reason() -> str | None:
    """Return why the split-layout contract cannot run here, or ``None``.

    The check is deliberately narrow. A missing Cargo, a Cargo too old to
    understand ``CARGO_BUILD_BUILD_DIR``, and a Linux host without ``ld.mold``
    each make the run impossible for a reason this contract does not own, so
    the run skips with that reason named. Any other failure surfaces from the
    commands themselves rather than being disguised as an unprovisioned host.
    """
    if shutil.which("cargo") is None:
        return "the split-layout contract needs cargo on the path"
    version = _cargo_version()
    if version < MINIMUM_CARGO:
        return (
            "the split-layout contract needs Cargo 1.91 or later, which "
            f"introduced CARGO_BUILD_BUILD_DIR; found {version[0]}.{version[1]}"
        )
    if sys.platform.startswith("linux") and shutil.which("ld.mold") is None:
        return (
            "the Linux build links with ld.mold, which make install-dev-fast "
            "installs"
        )
    return None


def split_build_environment(target: Path, build: Path) -> dict[str, str]:
    """Return an environment naming distinct target and build directories.

    Every inherited selection is removed first: a caller that already set one
    would otherwise make this contract repeat that layout, or build into a
    checkout's shared directory. ``HOME``, ``CARGO_HOME`` and ``RUSTUP_HOME``
    stay inherited so the repository's pinned toolchain and the shared crate
    cache are used rather than downloaded again.

    Quiet and verbose output are turned off rather than merely cleared: the
    suite assertion reads Cargo's ``Running tests/<suite>.rs`` lines, which an
    inherited or configured ``term.quiet`` would suppress and ``term.verbose``
    would replace with a compiler-style command line. The environment values
    are assigned because they outrank a user's Cargo configuration.
    """
    environment = os.environ.copy()
    for name in CLEARED_VARIABLES:
        environment.pop(name, None)
    environment["CARGO_TARGET_DIR"] = str(target)
    environment["CARGO_BUILD_BUILD_DIR"] = str(build)
    environment["CARGO_TERM_COLOR"] = "never"
    environment["CARGO_TERM_QUIET"] = "false"
    environment["CARGO_TERM_VERBOSE"] = "false"
    return environment


def run_cargo(
    environment: dict[str, str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    """Run one Cargo command from the repository root."""
    return subprocess.run(
        ["cargo", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def assert_suites_ran(completed: subprocess.CompletedProcess[str]) -> dict[str, Path]:
    """Require each selected suite to report a non-zero passing test count.

    Cargo prints one ``Running tests/<suite>.rs (<path>)`` line per selected
    test binary, and each libtest harness then prints one
    ``test result: ok.`` summary. Both are checked, because a run that
    selected no tests would otherwise pass against an empty result set. The
    harness paths are returned so the layout assertion can follow the profile
    directory Cargo actually used.
    """
    harnesses: dict[str, Path] = {}
    for suite in GATE_SUITES:
        # The capture is greedy and anchored to the line end, so a temporary
        # directory containing parentheses keeps its full path.
        match = re.search(
            rf"Running tests[\\/]{re.escape(suite)}\.rs \((.+)\)\s*$",
            completed.stderr,
            re.MULTILINE,
        )
        assert match is not None, (
            f"Cargo did not run the {suite} test binary:\n{completed.stderr}"
        )
        # Cargo prints paths relative to its working directory when it can.
        # Resolving against the repository root keeps those usable, while an
        # absolute capture is returned unchanged by the path join.
        harnesses[suite] = REPO_ROOT / Path(match.group(1))
    results = re.findall(r"test result: ok\. (\d+) passed", completed.stdout)
    assert len(results) == len(GATE_SUITES), (
        "expected one passing summary per gate suite, got "
        f"{results!r}:\n{completed.stdout}"
    )
    assert all(int(count) > 0 for count in results), (
        f"a gate suite ran no tests: {results!r}"
    )
    return harnesses


def assert_artefact_layout(
    target: Path, build: Path, harnesses: dict[str, Path]
) -> None:
    """Require finished binaries in ``target`` and hashes in ``build``.

    The suites would also run if Cargo ignored ``CARGO_BUILD_BUILD_DIR`` and
    put everything in one directory, so this is the assertion that the layout
    was actually split. The profile directory comes from the harness path
    Cargo printed rather than from an assumed ``debug``: a ``build.target``
    in Cargo configuration appends a target triple to both roots, and removing
    ``CARGO_BUILD_TARGET`` from the environment cannot undo that.
    """
    for suite, gate in GATE_SUITES.items():
        harness = harnesses[suite]
        assert harness.is_file(), f"the {suite} harness is missing: {harness}"
        assert HASHED_ARTEFACT.search(harness.name), (
            f"the {suite} harness is not a hashed artefact: {harness.name}"
        )
        relative_profile = harness.parent.parent.relative_to(build)
        finished = target / relative_profile / _executable(gate)
        assert finished.is_file(), (
            f"the finished {gate} binary is missing from {finished.parent}"
        )


def test_gate_binaries_run_when_cargo_splits_the_build_directory(
    tmp_path: Path,
) -> None:
    """Both gate suites must run for real with the build directory split out.

    ``cargo fetch --locked`` runs first because the suite command is offline:
    the contract must not turn a missing crate cache into a misleading layout
    failure. The test command then exercises exactly the two suites, so the
    summaries parsed below belong to them and not to unrelated package tests.
    """
    reason = skip_reason()
    if reason is not None:
        pytest.skip(reason)

    target = tmp_path / "target"
    build = tmp_path / "build"
    environment = split_build_environment(target, build)

    fetched = run_cargo(environment, "fetch", "--locked")
    assert fetched.returncode == 0, (
        f"cargo fetch --locked failed: {fetched.stderr}"
    )

    completed = run_cargo(
        environment,
        "test",
        "--offline",
        "--locked",
        "-p",
        "chutoro-test-support",
        "--all-features",
        "--test",
        "benchmark_regression_gate_cli",
    )
    assert completed.returncode == 0, (
        "the gate suites failed with Cargo's build directory split from its "
        f"target directory:\n{completed.stdout}\n{completed.stderr}"
    )
    assert "running 0 tests" not in completed.stdout
    assert "running 0 tests" not in completed.stderr
    harnesses = assert_suites_ran(completed)
    assert_artefact_layout(target, build, harnesses)