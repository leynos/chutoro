"""Property cases for the development-filter contract.

The contract's decisive assertions live in `dev_fast_filter_contract`, which is
a support module rather than a test module: the routing suite imports it, and
pytest only collects files named ``*_test.py``. A property written there would
therefore never run. The cases here are collected, so the property they carry
is executed by `make test-workflow-contracts`.

Two oracles are used, on purpose. `selection_of` reads the expression with an
independent parser, and `expected_development_filter` states the literal form
the recipe must pass. The exact-form oracle is the stronger of the two: the
selection oracle can only see tests its model knows, so an appended
``& not package(chutoro-providers-text)`` selects the intended set as modelled
while dropping a real package's tests in the workspace.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from dev_fast_expression_model import (
    DEV_PACKAGE,
    POISONED_LOCK_TEST,
    UNIVERSE,
    intended_selection,
    selection_of,
)
from dev_fast_filter_contract import (
    assert_development_exclusion,
    expected_development_filter,
    wider_development_filters,
)
from hypothesis import given, settings
from hypothesis import strategies as st

__all__ = [
    "test_contract_accepts_only_the_intended_expression",
    "test_contract_rejects_every_wider_filter",
]


def _compose_filter(
    exclusion: str,
    extra_clause: str | None,
    shape: str,
    negate_kind: bool,
) -> str:
    """Compose one filter in a named shape from generated pieces.

    ``negate_kind`` swaps ``kind(bench)`` for ``kind(test)``, which inverts
    what the leading clause removes without changing the exclusion's shape.
    That is what shows the contract is deciding on the whole expression rather
    than on its prefix.
    """
    kind = "kind(bench)" if negate_kind else "kind(test)"
    scoped = f"package({DEV_PACKAGE}) & test(={exclusion})"
    if shape == "appended":
        return f"not {kind} & not ({scoped}) & test(={extra_clause or exclusion})"
    if shape == "widened":
        return f"not {kind} & not ({scoped} | test(={extra_clause or exclusion}))"
    if shape == "dropped-package":
        return f"not {kind} & not (test(={exclusion}))"
    return f"not {kind} & not ({scoped})"


def _accepts(expression: str) -> bool:
    """Return whether the contract accepts ``expression``."""
    try:
        assert_development_exclusion(expression)
    except AssertionError:
        return False
    return True


#: Builds a filter from the pieces the real one is made of, so generated
#: expressions stay close enough to the documented grammar to model.
_FILTER_STRATEGIES: st.SearchStrategy[str] = st.builds(
    _compose_filter,
    exclusion=st.sampled_from(
        (
            POISONED_LOCK_TEST,
            UNIVERSE[1],
            "root_level_case",
        )
    ),
    extra_clause=st.one_of(st.none(), st.sampled_from(UNIVERSE)),
    shape=st.sampled_from(("exact", "appended", "widened", "dropped-package")),
    negate_kind=st.booleans(),
)


@settings(max_examples=200, deadline=None)
@given(expression=_FILTER_STRATEGIES)
def test_contract_accepts_only_the_intended_expression(expression: str) -> None:
    """Accept exactly the expected form; reject every generated near miss.

    Acceptance is decided against the literal expected filter, so a generated
    expression the tables never spell out still has a definite answer: it is
    accepted only when it *is* the intended filter. Anything the generator
    builds that differs — an appended conjunct, a widened disjunction, a
    dropped package clause, an inverted kind clause — must be rejected.

    The model answers the same question independently, and the two oracles are
    cross-checked. An expression the model says selects the intended set, but
    which is not the expected text, is exactly the gap this records: the
    ``extra-package`` near miss is such a case, and it must be rejected even
    though the modelled selection is right.
    """
    is_expected = expression == expected_development_filter()
    selects_intended = selection_of(expression) == intended_selection
    assert _accepts(expression) == is_expected, (
        "the contract and the expected form disagree about "
        f"{expression!r}: modelled selection {sorted(selection_of(expression))!r}"
    )
    assert is_expected <= selects_intended, (
        "an expression the contract accepts must also select the intended set, "
        f"but {expression!r} selects {sorted(selection_of(expression))!r}"
    )


def test_contract_rejects_every_wider_filter() -> None:
    """Each named near miss is rejected, with an evidence-bearing message.

    The table cases in the routing suite assert rejection through the recipe.
    This one asserts it directly on the contract, so a rejection that happens
    only because the recipe mangled the expression cannot hide a contract that
    would have accepted it.
    """
    assert_development_exclusion(expected_development_filter())
    for case, expression in wider_development_filters():
        assert not _accepts(expression), f"contract accepted the {case} filter"
