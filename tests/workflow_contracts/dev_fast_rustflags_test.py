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
from dataclasses import dataclass
from pathlib import Path

import pytest

from dev_fast_make_support import _make_output, _mutated_makefile
from dev_fast_rustflags_support import (
    _assert_build_rustflags,
    _assert_test_rustflags,
)

#: The route assertions, keyed by the target whose recipe they read. A mapping
#: rather than a branch per target keeps the mutation cases declarative.
_ROUTE_ASSERTIONS: dict[str, Callable[[str, str], None]] = {
    "dev-build": _assert_build_rustflags,
    "dev-test": _assert_test_rustflags,
}


@dataclass(frozen=True)
class _RouteMutation:
    """One targeted rewrite of a development recipe, with its own assertion.

    The three text fields and the assertion are always used together: the
    mutation is only meaningful against the recipe it was written for, and the
    defect it simulates is only detectable by that route's own contract. Keeping
    them in one object means the parametrized cases read as data instead of a
    five-argument tuple.
    """

    target: str
    anchor: str
    replacement: str

    @property
    def assertion(self) -> Callable[[str, str], None]:
        """Return the contract that owns this recipe."""
        return _ROUTE_ASSERTIONS[self.target]


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
    "mutation",
    (
        pytest.param(
            _RouteMutation(
                target="dev-build",
                anchor='RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(DEV_FRAGMENT_RUSTFLAGS)"',
                replacement='RUSTFLAGS="$(DEV_FRAGMENT_RUSTFLAGS)"',
            ),
            id="dev-build-replaces-inherited",
        ),
        pytest.param(
            _RouteMutation(
                target="dev-test",
                anchor='RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(DEV_TEST_RUSTFLAGS)"',
                replacement='RUSTFLAGS="$(DEV_TEST_RUSTFLAGS)"',
            ),
            id="dev-test-replaces-inherited",
        ),
    ),
)
def test_route_assertions_reject_replacing_inherited_rustflags(
    mutation: _RouteMutation, tmp_path: Path
) -> None:
    """A route that replaces `RUSTFLAGS` must fail its own contract.

    This is the regression the composition exists to prevent: writing only the
    route flags still passes every "lost flag" check, because those flags are
    present in the assignment. Only the expansion proves the inherited value
    survives, so a mutation that drops just the expansion must be rejected.
    """
    makefile = _mutated_makefile(tmp_path, mutation.anchor, mutation.replacement)
    output = _make_output(
        mutation.target, makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    with pytest.raises(AssertionError, match="inherited RUSTFLAGS"):
        mutation.assertion(output, "Linux")
