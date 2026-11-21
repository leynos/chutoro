"""The filter reading that keeps an invocation to one regression.

``selected_regression`` reads a nextest command's filters and refuses anything
other than exactly one exact-match expression, so a widened selection cannot
keep an invocation non-empty after its regression is renamed. Nextest accepts
a filter in several spellings and unions two filtersets, so the reader has to
see every spelling: one it skipped could widen the selection while the
expression it did recognise still looked right.

These cases live apart from ``scheduling_contract_test`` so the target-recipe
assertions and this reading keep their files inside the 400-line limit
``AGENTS.md`` sets.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from scheduling_reading import (
    QUALIFIED_REGRESSIONS,
    REGRESSIONS,
    selected_regression,
)


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param("--filterset=slow", id="a-named-filterset"),
        pytest.param("-E'all()'", id="an-attached-expression"),
    ],
)
def test_a_filter_the_selector_does_not_name_is_rejected(argument: str) -> None:
    """A filter beside the exact selector widens what the invocation runs.

    Nextest unions filtersets, so an attached ``-E`` value or a named set
    selects more tests than the one regression the invocation exists to run.
    The reading counts it, so the command is refused rather than accepted on
    the strength of the expression it happened to recognise.
    """
    command = (
        f"cargo nextest run {argument} -E "
        f"'test(={QUALIFIED_REGRESSIONS[REGRESSIONS[0]]})'"
    )
    with pytest.raises(ValueError, match="exactly one filter expression"):
        selected_regression(command)


def test_a_second_filter_argument_is_rejected() -> None:
    """A filterset beside the expression widens the selection.

    Nextest unions two filtersets, so ``--filterset`` beside ``-E`` selects
    more tests than the named regression while the exact filter is still
    there. The reading has to see the extra argument and refuse the command,
    not count the expression it recognised and ignore the rest.
    """
    widened = (
        "cargo nextest run --filterset=slow -E "
        f"'test(={QUALIFIED_REGRESSIONS[REGRESSIONS[0]]})'"
    )
    with pytest.raises(ValueError, match="exactly one filter expression"):
        selected_regression(widened)

def test_a_redirection_target_is_not_a_filter() -> None:
    """A file named like a filter expression selects nothing.

    ``> '-Etest(=...)'`` writes a file whose name looks like the selector.
    Reading it as a filter would accept a command that passes none, so the
    target has to be dropped before the filters are counted.
    """
    command = (
        "cargo nextest run > "
        f"'-Etest(={QUALIFIED_REGRESSIONS[REGRESSIONS[0]]})'"
    )
    with pytest.raises(ValueError, match="exactly one filter expression"):
        selected_regression(command)
