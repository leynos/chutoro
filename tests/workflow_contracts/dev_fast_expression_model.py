"""An independent model of what a nextest filter expression selects.

`dev_fast_filter_contract` decides whether the filter `make dev-test` passes to
nextest is the intended one. That decision is only as good as its reading of
the expression, so the reading lives here on its own: a small parser written
from the documented nextest grammar, over a fixed universe of tests wide enough
to tell the set shapes apart.

Kept separate from the contract on purpose. The contract's property compares
this model's answer with the contract's own verdict, and two readings that
shared code could not disagree - so this module never imports the contract.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

#: The case Cranelift cannot run: it aborts on a panic it cannot unwind, and
#: this regression poisons a lock by panicking under `catch_unwind`. Named here
#: rather than in the contract so the model can key its universe on it without
#: importing the thing it is used to check.
POISONED_LOCK_TEST = (
    "hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring"
)

#: A small universe wide enough to distinguish the set shapes the filter can
#: produce: the case under test, another case in the same package, a benchmark
#: in the same package, and a case in another package that shares the excluded
#: name. The last one is what makes the package clause observable: nextest's
#: `test(=name)` matches the bare name, so an unqualified exclusion would drop
#: the other crate's same-named case as well.
type TestName = str

DEV_PACKAGE = "chutoro-core"
OTHER_PACKAGE = "other-crate"

#: Package and kind are independent: the standard gate's benchmark lives in
#: the same package as ordinary cases, so a model that tied them together
#: would make `kind(bench)` and `package(...)` inseparable.
_PACKAGE_OF: dict[TestName, str] = {
    POISONED_LOCK_TEST: DEV_PACKAGE,
    "hnsw::cpu::other::case": DEV_PACKAGE,
    "bench::hnsw::throughput": DEV_PACKAGE,
    "other-crate::test::case": OTHER_PACKAGE,
    # Same bare test name, different package: unqualified exclusion drops it.
    POISONED_LOCK_TEST.rsplit("::", 1)[-1] + "@" + OTHER_PACKAGE: OTHER_PACKAGE,
}
BENCH_TESTS = frozenset({"bench::hnsw::throughput"})
UNIVERSE: tuple[TestName, ...] = tuple(_PACKAGE_OF)

#: What `dev-test` is meant to run: the standard selection, minus the one case.
intended_selection: frozenset[TestName] = frozenset(
    set(UNIVERSE) - BENCH_TESTS - {POISONED_LOCK_TEST}
)


def model_selection(expression: str) -> frozenset[TestName]:
    """Return the tests the model says `expression` selects.

    Written from the documented nextest grammar rather than by asking nextest,
    so the property below compares two independent readings. The grammar here
    is deliberately partial: it covers the operators the filter uses — ``not``,
    ``and``, ``or``, parentheses, and the ``kind``, ``package`` and ``test``
    predicates — and raises on anything else, so a filter written in an
    operator the model does not model fails the property rather than passing
    vacuously.

    >>> sorted(model_selection("not kind(bench) & package(chutoro-core) & not test(=hnsw::cpu::other::case)"))
    ['hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring']
    """
    tokens = _tokens(expression)
    selected, position = _parse_or(tokens, 0)
    assert position == len(tokens), (
        f"expression has trailing tokens at {position}: {expression!r}"
    )
    return frozenset(selected)


def _tokens(expression: str) -> tuple[str, ...]:
    """Split an expression into terms and operators, keeping predicates whole.

    A parenthesis belongs to a leading term rather than standing alone: it
    either completes a predicate, as in ``kind(bench)``, or opens a group.
    Splitting every parenthesis would break the predicates the model needs.

    >>> _tokens("not kind(bench) & not (test(=a))")
    ('not', 'kind(bench)', '&', 'not', '(', 'test(=a)', ')')
    """
    # A ``(`` straight after a name, with no space between, opens that
    # predicate's argument list and is part of it; any other ``(`` opens a
    # group and stands alone. That one rule keeps ``kind(bench)``,
    # ``package(chutoro-core)`` and ``test(=a)`` whole while leaving
    # ``not ( ... )`` as three tokens.
    tokens: list[str] = []
    index = 0
    while index < len(expression):
        character = expression[index]
        if character.isspace():
            index += 1
        elif character in "&|()":
            tokens.append(character)
            index += 1
        else:
            start = index
            while (
                index < len(expression)
                and not expression[index].isspace()
                and expression[index] not in "&|()"
            ):
                index += 1
            if index < len(expression) and expression[index] == "(":
                close = _matching_paren(expression, index)
                tokens.append(expression[start : close + 1])
                index = close + 1
            else:
                tokens.append(expression[start:index])
    return tuple(tokens)


def _matching_paren(expression: str, opening: int) -> int:
    """Return the index of the ``)`` matching the ``(`` at ``opening``."""
    depth = 0
    for index in range(opening, len(expression)):
        if expression[index] == "(":
            depth += 1
        elif expression[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError(f"unclosed parenthesis in {expression!r}")


def _parse_or(
    tokens: tuple[str, ...], position: int
) -> tuple[set[TestName], int]:
    """Parse a disjunction, the loosest-binding operator."""
    left, position = _parse_and(tokens, position)
    while position < len(tokens) and tokens[position] == "|":
        right, position = _parse_and(tokens, position + 1)
        left = left | right
    return left, position


def _parse_and(
    tokens: tuple[str, ...], position: int
) -> tuple[set[TestName], int]:
    """Parse a conjunction of possibly-negated terms."""
    left, position = _parse_term(tokens, position)
    while position < len(tokens) and tokens[position] == "&":
        right, position = _parse_term(tokens, position + 1)
        left = left & right
    return left, position


def _parse_term(
    tokens: tuple[str, ...], position: int
) -> tuple[set[TestName], int]:
    """Parse one predicate, optionally negated or parenthesised."""
    assert position < len(tokens), "expression ends where a term is expected"
    token = tokens[position]
    if token == "not":
        inner, position = _parse_term(tokens, position + 1)
        return set(UNIVERSE) - inner, position
    if token == "(":
        inner, position = _parse_or(tokens, position + 1)
        assert position < len(tokens) and tokens[position] == ")", (
            "unclosed group in expression"
        )
        return inner, position + 1
    return _predicate(token), position + 1


def _predicate(token: str) -> set[TestName]:
    """Return the tests one leaf predicate selects."""
    if token == "kind(bench)":
        return set(BENCH_TESTS)
    if token == "kind(test)":
        return set(UNIVERSE) - BENCH_TESTS
    if token.startswith("package(") and token.endswith(")"):
        package = token[len("package(") : -1]
        return {name for name, owner in _PACKAGE_OF.items() if owner == package}
    assert token.startswith("test(=") and token.endswith(")"), (
        f"model does not cover this predicate: {token!r}"
    )
    # nextest matches the bare test name, so a class-qualified model key
    # named like the excluded case is matched too. That is what makes an
    # unqualified exclusion observably wider than a package-scoped one.
    named = token[len("test(=") : -1].rsplit("::", 1)[-1]
    return {name for name in UNIVERSE if bare(name) == named}


def bare(name: TestName) -> str:
    """Return the test name nextest would match, without the model's key suffix.

    >>> bare("case_name@other-crate")
    'case_name'
    >>> bare("hnsw::cpu::other::case")
    'case'
    """
    return name.split("@", 1)[0].rsplit("::", 1)[-1]


def selection_of(expression: str) -> frozenset[TestName]:
    """Return the model selection, extending the universe to any unknown name.

    A test name outside `UNIVERSE` would otherwise select nothing, making a
    filter that names an unrelated case look identical to the intended one.
    The model therefore adds every name the expression mentions, and treats
    each as an ordinary non-benchmark case in its own package.
    """
    return model_selection(expression) | _unknown_tests(expression)


def _unknown_tests(expression: str) -> frozenset[TestName]:
    """Return the mentioned test names the fixed universe does not hold."""
    return _named_tests(expression) - frozenset(UNIVERSE)


def _named_tests(expression: str) -> frozenset[TestName]:
    """Return every test name the expression mentions, known or not."""
    return frozenset(
        token[len("test(=") : -1]
        for token in _tokens(expression)
        if token.startswith("test(=") and token.endswith(")")
    )


