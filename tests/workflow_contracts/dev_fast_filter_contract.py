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
comparison would otherwise catch. Equality against the expected form rejects
all of them without needing to model which packages exist.

The `given` property that exercises this over a generated grammar lives in
`dev_fast_filter_contract_test`, not here. pytest collects files named
``*_test.py``, so a property in this support module would never run.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from dev_fast_expression_model import (
    POISONED_LOCK_TEST,
    intended_selection,
    selection_of,
)

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

    The whole expression is compared last, and compared for equality rather
    than searched for a substring. The exclusion is one group inside a negated
    clause, so a substring check accepts a filter that adds a second exclusion
    — ``... & test(=other)`` outside the group, or ``... | test(=other)``
    inside it — and also one that drops the package clause instead of adding
    anything. Each silently narrows `dev-test` past the single case it is
    meant to give up, which is the same class of omission the selection
    contract exists to catch. Only the end-to-end comparison would notice
    otherwise, and that is opt-in, so it never runs on CI.

    Equality against the expected form is what makes this complete. A
    selection comparison alone is not: it can only see tests the model knows,
    so ``& not package(chutoro-providers-text)`` selects the intended set *as
    modelled* while dropping a real package's tests in the workspace. The
    model would have to enumerate every package to catch that, and nextest's
    filter language is not closed over a fixed universe. Comparing the text
    needs no such enumeration, and it also rejects operators the model does
    not implement, such as ``-E``'s ``rdeps`` or ``default``, which would
    otherwise be read as selecting nothing.

    The model is not silent about an unmodelled crate — `_package_selection`
    rejects one — but that is a rejection the *model* raises, and it would
    fire on any expression naming a crate the model lacks. The comparison
    below is what constrains this filter, whatever the model does.
    """
    expected = expected_development_filter()
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
    try:
        selection = selection_of(expression)
    except LookupError as unmodelled:
        raise AssertionError(
            f"dev-test filter names a crate the model cannot place: {unmodelled}"
        ) from unmodelled
    assert selection == intended_selection, (
        "dev-test must select the standard set minus the poisoned-lock case: "
        f"{expression!r} selects {sorted(selection)!r}"
    )
    assert expression == expected, (
        "dev-test must pass exactly the intended filter; "
        f"expected {expected!r}, got {expression!r}"
    )


def wider_development_filters() -> tuple[tuple[str, str], ...]:
    """Return filters that each exclude more than the single intended case.

    Every entry keeps the poisoned-lock clause and the expected prefix, so
    each one passes a presence-and-prefix check; they differ only in how they
    admit more exclusion, which is what the exact comparison must catch.

    The two ``*-package`` cases name crates the model holds no entry for, so
    each drops a real crate's tests in the workspace while the modelled set is
    unchanged. They are rejected twice over: the model raises on an unmodelled
    crate, and the exact-form comparison rejects the text regardless.

    >>> [(case, expr != expected_development_filter())
    ...  for case, expr in wider_development_filters()]
    [('appended-clause', True), ('widened-group', True), ('dropped-package', True), ('extra-package', True), ('unmodelled-package', True)]
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
        ("extra-package", f"{expected} & not package(chutoro-providers-text)"),
        ("unmodelled-package", f"{expected} & not package(chutoro-providers-dense)"),
    )
