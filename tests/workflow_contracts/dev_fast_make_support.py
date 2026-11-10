"""Evaluation helpers shared by the development-route contract suites.

The development targets are Make recipes, so the only honest way to check what
they run is to ask `make` to evaluate them and read the result. These helpers
read a recipe's Cargo invocations out of a dry run, locate the pinned fragment,
and assert the routing properties every suite needs. They live in a support
module rather than a test module so several suites can share them without
importing one test suite from another.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE_PATH = REPO_ROOT / "Makefile"
DEV_FAST_TOOLCHAIN_PATH = REPO_ROOT / "tools" / "dev-fast" / "TOOLCHAIN"
DEV_FAST_CONFIG = "tools/dev-fast/config.toml"


def _make_output(
    target: str,
    *,
    overrides: tuple[str, ...] = (),
    makefile: Path = MAKEFILE_PATH,
    jobs: int | None = None,
) -> str:
    """Return evaluated Make recipes for one target without executing them.

    ``--always-make`` is what makes the reading independent of the checkout:
    without it a target whose file already exists prints nothing and the
    contract would silently pass over an empty script.
    """
    command = ["make", "--no-print-directory", "--dry-run", "--always-make"]
    if jobs is not None:
        command.append(f"-j{jobs}")
    command.extend(("--file", str(makefile), *overrides, target, "CARGO=probe-cargo"))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"make dry-run for {target!r} failed with {completed.returncode}: "
        f"{completed.stderr}"
    )
    return completed.stdout


def _cargo_invocations(output: str) -> list[tuple[str, list[str]]]:
    """Return each evaluated probe-cargo recipe with its shell tokens.

    A recipe may chain several Cargo commands on one line - `make test` runs
    `$(CARGO) fetch --locked && ... $(CARGO) nextest run ...` - so every
    occurrence starts a new invocation and ends where the next one begins.
    Reading the line as a single command would report `fetch` as the verb and
    hide the nextest run behind it.
    """
    invocations: list[tuple[str, list[str]]] = []
    for line in output.splitlines():
        tokens = shlex.split(line)
        starts = [index for index, token in enumerate(tokens) if token == "probe-cargo"]
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(tokens)
            invocations.append((line, tokens[start:end]))
    return invocations


def _pinned_toolchain() -> str:
    """Return the toolchain channel pinned for the development route."""
    toolchain = DEV_FAST_TOOLCHAIN_PATH.read_text(encoding="utf-8").strip()
    assert toolchain, f"{DEV_FAST_TOOLCHAIN_PATH} must contain a toolchain pin"
    return toolchain


def _installed_toolchains() -> set[str] | None:
    """Return the toolchain names rustup reports, or ``None`` if it failed.

    ``rustup toolchain list`` prints one name per line, usually with a host
    triple appended and optionally a ``(default)`` or ``(active)`` marker.
    Splitting each line on whitespace keeps the name and drops the marker, so
    callers can compare names rather than substrings.

    A host without rustup at all has no installed toolchains, which is the
    same answer as a rustup that ran and failed. Both must report ``None``
    rather than raise, or the skip guard would turn an absent toolchain into
    an error.
    """
    try:
        listed = subprocess.run(
            ["rustup", "toolchain", "list"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if listed.returncode != 0:
        return None
    return {
        fields[0] for fields in map(str.split, listed.stdout.splitlines()) if fields
    }


def _is_toolchain_installed(toolchain: str, installed: set[str]) -> bool:
    """Report whether a pin is installed as itself or with a host triple.

    A substring test would also match a longer name that merely contains the
    pin, so a host holding only ``nightly-2026-05-28-something`` would skip
    a real Cargo check while reporting the pin as present. Accepting the
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


class ComponentProbeError(RuntimeError):
    """The component listing could not be read, so its answer is unknown."""


def _installed_components(toolchain: str) -> tuple[str, ...]:
    """Return the components rustup reports for a toolchain.

    A listing that cannot be read raises :class:`ComponentProbeError` rather
    than returning an empty tuple, because "rustup is broken or absent" and
    "the component is not installed" are different facts: the first is a
    defect in the probe's host, the second is a host that has simply not run
    ``make install-dev-fast``.
    """
    try:
        listed = subprocess.run(
            ["rustup", "component", "list", "--installed", "--toolchain", toolchain],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise ComponentProbeError(
            f"rustup is not installed, so {toolchain}'s components cannot be read"
        ) from error
    if listed.returncode != 0:
        raise ComponentProbeError(
            f"rustup could not list {toolchain}'s components: {listed.stderr.strip()}"
        )
    return tuple(map(str.strip, listed.stdout.splitlines()))


def _has_cranelift_backend(toolchain: str) -> bool:
    """Report whether a toolchain carries the Cranelift codegen component.

    The component is a separate rustup install, so a channel that is present
    can still refuse ``-Z codegen-backend=cranelift`` with "failed to find a
    ``codegen-backends`` folder". CI hits exactly that: it installs the same
    nightly pin for the Whitaker suite, but only the stable components.

    ``rustup component list --installed`` names each component with a host
    triple appended, so the bare name is matched as a prefix. Output is
    matched per line rather than by substring, so ``rustc-codegen-cranelift``
    cannot be confused with a longer component name.

    An unreadable listing raises :class:`ComponentProbeError`; only a reading
    that succeeded can say the component is absent.
    """
    components = _installed_components(toolchain)
    return any(
        name == "rustc-codegen-cranelift"
        or name.startswith("rustc-codegen-cranelift-")
        for name in components
    )


def _skip_unless_pin_installed(reason: str, *, needs_cranelift: bool = False) -> None:
    """Skip the calling test when the pinned nightly is not provisioned.

    Provisioning it is ``make install-dev-fast``'s job, so a host that has not
    done so must skip rather than reach the network. ``reason`` names what
    else the caller needs, so a skip says what is missing rather than
    implying the route is broken.

    ``needs_cranelift`` narrows the check to callers that actually compile.
    Presence of the channel alone is not enough for them: CI installs the same
    nightly pin for the Whitaker suite without the Cranelift component, so a
    toolchain-only guard would run the build and then fail on the missing
    backend. Callers that only read configuration leave it ``False``.

    A listing that cannot be read is not a skip: the host claims the toolchain
    is installed, so an unreadable component list is a defect in the probe or
    in rustup, and :class:`ComponentProbeError` reaches the test boundary with
    that fact rather than being reported as an unprovisioned host.
    """
    toolchain = _pinned_toolchain()
    installed = _installed_toolchains()
    if installed is None or not _is_toolchain_installed(toolchain, installed):
        pytest.skip(
            f"the pinned development toolchain {toolchain} is not installed; "
            f"{reason}"
        )
    if needs_cranelift and not _has_cranelift_backend(toolchain):
        pytest.skip(
            f"the pinned toolchain {toolchain} lacks rustc-codegen-cranelift; "
            f"{reason}"
        )


def _assert_routed_cargo(output: str, expected_verbs: tuple[str, ...]) -> None:
    """Require every emitted Cargo command to select the pinned fragment."""
    toolchain = _pinned_toolchain()
    assert (REPO_ROOT / DEV_FAST_CONFIG).is_file(), (
        f"the selected Cargo fragment is missing: {DEV_FAST_CONFIG}"
    )

    invocations = _cargo_invocations(output)
    assert len(invocations) == len(expected_verbs), (
        f"expected Cargo verbs {expected_verbs!r}, found "
        f"{[tokens for _, tokens in invocations]!r}"
    )
    for (line, tokens), verb in zip(invocations, expected_verbs, strict=True):
        expected_prefix = [f"+{toolchain}", "--config", DEV_FAST_CONFIG, verb]
        assert tokens[1 : 1 + len(expected_prefix)] == expected_prefix, (
            f"every development Cargo invocation must begin with "
            f"'probe-cargo {' '.join(expected_prefix)}'; got {line!r}"
        )


def _assert_unfragmented(output: str, target: str) -> None:
    """Require a standard target to stay off the development fragment."""
    assert DEV_FAST_CONFIG not in output, (
        f"{target} must not select {DEV_FAST_CONFIG}; got:\n{output}"
    )


def _assert_nextest_selects_fragment(line: str, tokens: list[str]) -> None:
    """Require nextest's inner Cargo invocation to select the fragment."""
    nextest_args = tokens[tokens.index("nextest") + 1 :]
    profile_index = nextest_args.index("--profile")
    config_indices = [
        index for index, token in enumerate(nextest_args) if token == "--config"
    ]
    assert (
        len(config_indices) == 1
        and config_indices[0] < profile_index
        and nextest_args[config_indices[0] + 1] == DEV_FAST_CONFIG
    ), (
        f"nextest must pass exactly one {DEV_FAST_CONFIG} selection "
        f"before --profile: {line!r}"
    )


def _assert_nextest_scope(line: str, tokens: list[str]) -> None:
    """Check the development run covers the same suite as the gate.

    ``dev-test`` is a convenience, not evidence for merge, but it is worthless
    if it silently narrows the suite. Dropping ``--all-targets`` or
    ``--all-features`` would skip tests, ``--all-targets`` alone is not enough
    without ``run``, and omitting the benchmark filter makes nextest attempt
    cases the standard gate excludes.
    """
    nextest_args = tokens[tokens.index("nextest") + 1 :]
    assert nextest_args[0] == "run", (
        f"dev-test must invoke 'nextest run', got {nextest_args[0]!r}: {line!r}"
    )
    for required in ("--all-targets", "--all-features"):
        assert required in nextest_args, (
            f"dev-test must pass {required} so the suite matches the gate: {line!r}"
        )
    assert "-E" in nextest_args, f"dev-test must filter benchmarks: {line!r}"
    filter_index = nextest_args.index("-E")
    assert nextest_args[filter_index + 1] == "not kind(bench)", (
        f"dev-test must exclude only benchmarks: {line!r}"
    )


def _mutated_makefile(tmp_path: Path, old: str, new: str) -> Path:
    """Write a private Makefile copy after one asserted text mutation."""
    source = MAKEFILE_PATH.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"mutation anchor must occur once: {old!r}"
    mutated = tmp_path / "Makefile"
    mutated.write_text(source.replace(old, new), encoding="utf-8")
    return mutated
