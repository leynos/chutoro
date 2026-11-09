"""Contract tests for `RUSTFLAGS` composition on the development routes.

Both development recipes assign `RUSTFLAGS`, and Cargo treats an assignment as
replacing every `rustflags` table - including the fragment's own. These
contracts pin the composition that keeps an inherited value while still
restating the fragment's mold selection and documentation denials.

The evaluation helpers live in ``dev_fast_rustflags_support`` so the routing
suite can assert the same policy without importing a test module.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from dev_fast_make_support import _make_output, _mutated_makefile
from dev_fast_rustflags_support import (
    _assert_build_rustflags,
    _assert_test_rustflags,
)


@pytest.mark.parametrize("host_os", ("Linux", "Darwin"))
def test_development_build_route_restates_the_fragment_flags(host_os: str) -> None:
    """`dev-build` must keep the fragment's flags when `RUSTFLAGS` is inherited."""
    output = _make_output("dev-build", overrides=(f"DEV_FAST_HOST_OS={host_os}",))
    _assert_build_rustflags(output, host_os)


@pytest.mark.parametrize("host_os", ("Linux", "Darwin"))
def test_development_test_route_preserves_warning_and_platform_flags(
    host_os: str,
) -> None:
    """`dev-test` must restate the denials, and the Linux linker, itself.

    An assigned `RUSTFLAGS` replaces the fragment's `rustflags` table, so the
    route would silently lose both if it did not repeat them.
    """
    output = _make_output("dev-test", overrides=(f"DEV_FAST_HOST_OS={host_os}",))
    _assert_test_rustflags(output, host_os)


def test_test_flag_assertion_rejects_a_missing_denial(tmp_path: Path) -> None:
    """The effective development environment must retain every denial."""
    makefile = _mutated_makefile(
        tmp_path,
        "DEV_FRAGMENT_RUSTFLAGS = $(strip -Dmissing_docs",
        "DEV_FRAGMENT_RUSTFLAGS = $(strip",
    )
    output = _make_output(
        "dev-test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    with pytest.raises(AssertionError, match="lost"):
        _assert_test_rustflags(output, "Linux")


@pytest.mark.parametrize(
    ("target", "anchor", "replacement", "assertion"),
    (
        pytest.param(
            "dev-build",
            'RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(DEV_FRAGMENT_RUSTFLAGS)"',
            'RUSTFLAGS="$(DEV_FRAGMENT_RUSTFLAGS)"',
            _assert_build_rustflags,
            id="dev-build-replaces-inherited",
        ),
        pytest.param(
            "dev-test",
            'RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(DEV_TEST_RUSTFLAGS)"',
            'RUSTFLAGS="$(DEV_TEST_RUSTFLAGS)"',
            _assert_test_rustflags,
            id="dev-test-replaces-inherited",
        ),
    ),
)
def test_route_assertions_reject_replacing_inherited_rustflags(
    target: str,
    anchor: str,
    replacement: str,
    assertion: Callable[[str, str], None],
    tmp_path: Path,
) -> None:
    """A route that replaces `RUSTFLAGS` must fail its own contract.

    This is the regression the composition exists to prevent: writing only the
    route flags still passes every "lost flag" check, because those flags are
    present in the assignment. Only the expansion proves the inherited value
    survives, so a mutation that drops just the expansion must be rejected.
    """
    makefile = _mutated_makefile(tmp_path, anchor, replacement)
    output = _make_output(
        target, makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    with pytest.raises(AssertionError, match="inherited RUSTFLAGS"):
        assertion(output, "Linux")
