"""Evaluation helpers for the development routes' `RUSTFLAGS` contracts.

The two development recipes assign `RUSTFLAGS`, and Cargo treats an assignment
as replacing every `rustflags` table - including the fragment's own. These
helpers read the evaluated `make` recipe and report whether a route kept the
inherited value while restating the fragment's flags. Keeping them in a support
module lets both the routing suite and the `RUSTFLAGS` suite assert the same
policy without importing one test module from another.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import shlex

from dev_fast_make_support import (
    DEV_FAST_CONFIG,
    _assert_nextest_scope,
    _assert_nextest_selects_fragment,
    _cargo_invocations,
    _pinned_toolchain,
)

#: The flags the development fragment's Linux table carries, restated by both
#: routes because an assigned `RUSTFLAGS` replaces that table outright.
REQUIRED_FRAGMENT_FLAGS = ("-Dmissing_docs", "-Dmissing_crate_level_docs")
REQUIRED_TEST_DENIALS = ("-D warnings", *REQUIRED_FRAGMENT_FLAGS)


def _assert_composed_with_inherited_rustflags(rustflags: str, target: str) -> None:
    """Require the route flags to be appended to an inherited `RUSTFLAGS`.

    Assigning `RUSTFLAGS` makes Cargo ignore the fragment's `rustflags` table,
    so a recipe that wrote only its own flags would drop the mold selection and
    both documentation denials whenever a developer had exported `RUSTFLAGS`.
    The parameter expansion is the behaviour under test: it must appear as the
    recipe's first token so the inherited value is preserved ahead of the route
    flags, and it must supply the separating space itself.
    """
    assert rustflags.startswith("${RUSTFLAGS:+$RUSTFLAGS }"), (
        f"{target} must append its flags to an inherited RUSTFLAGS instead of "
        f"replacing it, got {rustflags!r}"
    )


def _single_rustflags_assignment(line: str, target: str) -> str:
    """Return the one `RUSTFLAGS` value a development recipe assigns."""
    assignments = [
        token for token in shlex.split(line) if token.startswith("RUSTFLAGS=")
    ]
    assert len(assignments) == 1, f"{target} must set RUSTFLAGS once: {line!r}"
    return assignments[0].partition("=")[2]


def _assert_platform_linker(rustflags: str, host_os: str, target: str) -> None:
    """Require the mold flag exactly on Linux and nowhere else."""
    has_mold = "-Clink-arg=-fuse-ld=mold" in rustflags
    assert has_mold is (host_os == "Linux"), (
        f"{target} RUSTFLAGS must select mold only on Linux; "
        f"DEV_FAST_HOST_OS={host_os!r}, got {rustflags!r}"
    )


def _single_nextest_invocation(output: str) -> tuple[str, list[str]]:
    """Return the one Cargo invocation that runs the development test route."""
    invocations = [
        (line, tokens)
        for line, tokens in _cargo_invocations(output)
        if "nextest" in tokens
    ]
    assert len(invocations) == 1, (
        f"expected one development nextest invocation, found {invocations!r}"
    )
    return invocations[0]


def _assert_test_rustflags(output: str, host_os: str) -> None:
    """Check the development nextest run keeps the warning and linker policy."""
    line, tokens = _single_nextest_invocation(output)
    toolchain = _pinned_toolchain()
    assert tokens[1:5] == [
        f"+{toolchain}",
        "--config",
        DEV_FAST_CONFIG,
        "nextest",
    ], f"dev-test must use the selected route: {line!r}"
    _assert_nextest_selects_fragment(line, tokens)
    _assert_nextest_scope(line, tokens)
    rustflags = _single_rustflags_assignment(line, "dev-test")
    _assert_composed_with_inherited_rustflags(rustflags, "dev-test")
    missing = [flag for flag in REQUIRED_TEST_DENIALS if flag not in rustflags]
    assert not missing, f"dev-test RUSTFLAGS lost {missing!r}: {line!r}"
    _assert_platform_linker(rustflags, host_os, "dev-test")


def _assert_build_rustflags(output: str, host_os: str) -> None:
    """Check `dev-build` states the fragment's flags as an assignment.

    The build route sets no flags of its own, so without an assignment it reads
    the fragment's `rustflags` table. That is correct, but it turns into a
    silent loss the moment a developer exports `RUSTFLAGS`, because an assigned
    value replaces every table. Restating the table's flags preserves the mold
    selection and both documentation denials under either condition.
    """
    invocations = [
        (line, tokens) for line, tokens in _cargo_invocations(output) if "build" in tokens
    ]
    assert len(invocations) == 1, (
        f"expected one development build invocation, found {invocations!r}"
    )
    line, _ = invocations[0]
    rustflags = _single_rustflags_assignment(line, "dev-build")
    _assert_composed_with_inherited_rustflags(rustflags, "dev-build")
    missing = [flag for flag in REQUIRED_FRAGMENT_FLAGS if flag not in rustflags]
    assert not missing, f"dev-build RUSTFLAGS lost {missing!r}: {line!r}"
    _assert_platform_linker(rustflags, host_os, "dev-build")
