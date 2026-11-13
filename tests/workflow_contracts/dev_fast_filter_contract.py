"""The exact shape of the filter `make dev-test` hands to nextest.

`dev_fast_routing_test` reads the evaluated recipe and compares the result
here. Keeping the expected form and its near misses in one module means the
contract and the mutations that must fail it are stated together, and it keeps
the routing suite under the 400-line limit `AGENTS.md` sets.

The comparison is by whole expression, not substring. The exclusion is a
single group inside a negated clause, so a substring check accepts a filter
that adds a second exclusion outside the group, widens the group with ``|``,
or drops the package clause entirely. Each narrows `dev-test` past the one
case it is meant to give up, which only the opt-in end-to-end selection
comparison would otherwise catch.

The `given` property below states that invariant over a generated grammar
rather than a handful of cases, and decides it against
`dev_fast_expression_model`'s independent reading of the same expression: the
property accepts an expression if and only if the model says it selects the
standard set minus the one case.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from collections.abc import Callable

from dev_fast_expression_model import (
    DEV_PACKAGE,
    POISONED_LOCK_TEST,
    UNIVERSE,
    intended_selection,
    selection_of,
)
from hypothesis import given, settings
from hypothesis import strategies as st

__all__ = [
    "POISONED_LOCK_TEST",
    "assert_development_exclusion",
    "expected_development_filter",
    "wider_development_filters",
]


def expected_development_filter() -> str:
    """Return the one filter `make dev-test` should pass to nextest.

    Built from `POISONED_LOCK_TEST` rather than restated, so the expected form
    tracks a renamed case and cannot drift from the name the routing suite
    checks against the Rust source.

    >>> expected_development_filter()
    'not kind(bench) & not (package(chutoro-core) & test(=hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring))'
    """
    return (
        "not kind(bench) & not "
        f"(package(chutoro-core) & test(={POISONED_LOCK_TEST}))"
    )


def _compose_filter(
    exclusion: str,
    extra_clause: str | None,
    shape: str,
    negate_kind: bool,
) -> str:
    """Compose one filter in a named shape from generated pieces.

    ``negate_kind`` swaps ``kind(bench)`` for ``kind(test)``, which inverts
    what the leading clause removes without changing the exclusion's shape.
    That is what shows the property is deciding on the modelled selection
    rather than on the expression's prefix.
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
def test_contract_accepts_exactly_the_filters_that_select_the_intended_set(
    expression: str,
) -> None:
    """Accept only the expressions whose modelled selection is the intended one.

    The model is a second, independent reading of the documented grammar, so
    a generated combination the table cases never spell out still has a
    definite expected answer. Acceptance is compared to that answer, not to
    the shape, so an expression written another way but selecting the same
    tests is accepted and one written to look right but selecting fewer is
    rejected.

    The generated filter names only cases the universe holds, so the answer is
    always ``intended_selection``: every shape that removes more than the one
    case selects a strict subset and must be rejected. A name outside the
    universe reaches `selection_of`, which adds it to the modelled selection
    rather than ignoring it — an exclusion naming a case that does not exist
    removes nothing, and the property must see that rather than read the
    filter as intended.
    """
    selected = selection_of(expression)
    assert _accepts(expression) == (selected == intended_selection), (
        "the contract and the selection model disagree about "
        f"{expression!r}: modelled selection {sorted(selected)!r}"
    )


def assert_development_exclusion(expression: str) -> None:
    """Require the development filter to exclude exactly the poisoned-lock case.

    A missing close parenthesis is the failure mode this guards against: the
    expression still *reads* correctly, but nextest rejects it outright with
    `error: expected close parenthesis`, so `make dev-test` runs nothing at
    all. Balance is therefore checked before the textual assertions, which
    would otherwise pass against an unparseable expression.

    The polarity is checked as well as the presence. A negated case clause
    — ``... & not test(=...)`` — keeps every string the positive filter
    contains and still starts with the expected prefix, so a presence-only
    check accepts it. What `make dev-test` would then run is the
    poisoned-lock regression and nothing else, which is the opposite of the
    intended exclusion.

    The whole expression is compared last, not searched for a substring. The
    exclusion is one group inside a negated clause, so a substring check
    accepts a filter that adds a second exclusion — ``... & test(=other)``
    outside the group, or ``... | test(=other)`` inside it — and also one
    that drops the package clause instead of adding anything. Each silently
    narrows `dev-test` past the single case it is meant to give up, which is
    the same class of omission the selection contract exists to catch. Only
    the end-to-end comparison would notice otherwise, and that is opt-in, so
    it never runs on CI.
    """
    depth = 0
    for character in expression:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            assert depth >= 0, f"dev-test filter closes an unopened group: {expression!r}"
    assert depth == 0, (
        f"dev-test filter leaves {depth} unclosed group(s), which nextest "
        f"rejects: {expression!r}"
    )
    assert f"test(={POISONED_LOCK_TEST})" in expression, (
        f"dev-test must exclude the poisoned-lock regression: {expression!r}"
    )
    assert expression.startswith("not kind(bench) & not ("), (
        f"dev-test must filter benchmarks and then one named case: {expression!r}"
    )
    assert f"not test(={POISONED_LOCK_TEST})" not in expression, (
        "dev-test must exclude the poisoned-lock regression, not select it "
        f"alone: {expression!r}"
    )
    assert selection_of(expression) == intended_selection, (
        "dev-test must select the standard set minus the poisoned-lock case: "
        f"{expression!r} selects {sorted(selection_of(expression))!r}"
    )


def wider_development_filters() -> tuple[tuple[str, str], ...]:
    """Return filters that each exclude more than the single intended case.

    Every entry keeps the poisoned-lock clause and the expected prefix, so
    each one passes a presence-and-prefix check; they differ only in how they
    admit more exclusion, which is what the exact comparison must catch.

    >>> [(case, expr != expected_development_filter())
    ...  for case, expr in wider_development_filters()]
    [('appended-clause', True), ('widened-group', True), ('dropped-package', True)]
    """
    expected = expected_development_filter()
    return (
        ("appended-clause", f"{expected} & test(=other)"),
        (
            "widened-group",
            expected.replace(
                f"test(={POISONED_LOCK_TEST}))",
                f"test(={POISONED_LOCK_TEST}) | test(=other))",
            ),
        ),
        ("dropped-package", f"not kind(bench) & not (test(={POISONED_LOCK_TEST}))"),
    )
