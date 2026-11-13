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

#: A complete registered test name, as nextest reports it.
type TestName = str

#: A test as (package, complete registered name). A name alone is not a key:
#: the same name can be registered in two crates, and the model must be able to
#: hold both to show what ``package(...)`` is for.
type TestId = tuple[str, TestName]

#: A small universe wide enough to distinguish the set shapes the filter can
#: produce. Package and kind are independent: the benchmark lives in the same
#: package as ordinary cases, so a model that tied them together would make
#: `kind(bench)` and `package(...)` inseparable.
#:
#: ``COLLIDING_TEST`` is registered in both packages on purpose. nextest matches
#: ``test(=name)`` against the complete name, so one predicate matches both
#: registrations, and only a package clause separates them. That is what makes
#: the package clause observable rather than merely present.
DEV_PACKAGE = "chutoro-core"
OTHER_PACKAGE = "other-crate"
COLLIDING_TEST = "gw::collide::case"

#: A stand-in package for a test name the model has never seen, so an unknown
#: name is still one ordinary non-benchmark case and not an empty set.
_UNMODELLED_PACKAGE = "unmodelled"

UNIVERSE: tuple[TestId, ...] = (
    (DEV_PACKAGE, POISONED_LOCK_TEST),
    (DEV_PACKAGE, "hnsw::cpu::other::case"),
    (DEV_PACKAGE, "bench::hnsw::throughput"),
    (OTHER_PACKAGE, "other-crate::test::case"),
    (DEV_PACKAGE, COLLIDING_TEST),
    (OTHER_PACKAGE, COLLIDING_TEST),
)
BENCH_TESTS: frozenset[TestId] = frozenset({(DEV_PACKAGE, "bench::hnsw::throughput")})

#: What `dev-test` is meant to run: the standard selection, minus the one case.
intended_selection: frozenset[TestId] = frozenset(
    set(UNIVERSE) - BENCH_TESTS - {(DEV_PACKAGE, POISONED_LOCK_TEST)}
)


def model_selection(expression: str) -> frozenset[TestId]:
    """Return the (package, name) tests `expression` selects.

    Written from the documented nextest grammar rather than by asking nextest,
    so the property below compares two independent readings. The grammar here
    is deliberately partial: it covers the operators the filter uses — ``not``,
    ``and``, ``or``, parentheses, and the ``kind``, ``package`` and ``test``
    predicates — and raises on anything else, so a filter written in an
    operator the model does not model fails the property rather than passing
    vacuously.

    >>> sorted(model_selection("not kind(bench) & package(chutoro-core) & not test(=hnsw::cpu::other::case)"))
    [('chutoro-core', 'gw::collide::case'), ('chutoro-core', 'hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring')]

    The collision is the point of keying on ``(package, name)``: one
    ``test(=...)`` predicate matches both registrations of `COLLIDING_TEST`,
    and only the package clause narrows that to one crate's copy.

    >>> sorted(model_selection("test(=gw::collide::case)"))
    [('chutoro-core', 'gw::collide::case'), ('other-crate', 'gw::collide::case')]
    >>> sorted(model_selection("package(chutoro-core) & test(=gw::collide::case)"))
    [('chutoro-core', 'gw::collide::case')]
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
    cursor = _Cursor(expression)
    tokens: list[str] = []
    while not cursor.is_done:
        tokens.append(cursor.next_token())
    return tuple(tokens)


class _Cursor:
    """Walk an expression, yielding one whole token per step."""

    def __init__(self, expression: str) -> None:
        """Start the cursor at the first character of ``expression``."""
        self._expression = expression
        self._index = 0

    @property
    def is_done(self) -> bool:
        """Return whether every character has been consumed."""
        return self._index >= len(self._expression)

    def next_token(self) -> str:
        """Return the token at the cursor and advance past it."""
        while not self.is_done and self._expression[self._index].isspace():
            self._index += 1
        assert not self.is_done, "next_token called with nothing left"
        if self._expression[self._index] in "&|()":
            return self._advance(1)
        return self._next_named_token()

    def _advance(self, count: int) -> str:
        """Return the ``count`` characters at the cursor and step past them."""
        start = self._index
        self._index += count
        return self._expression[start : self._index]

    def _next_named_token(self) -> str:
        """Return a name, with its argument list when one follows it."""
        start = self._index
        self._scan_name()
        self._scan_argument_list_if_present()
        return self._expression[start : self._index]

    def _scan_name(self) -> None:
        """Advance over the run of non-separator characters at the cursor."""
        while not self.is_done and not self._is_separator(self._expression[self._index]):
            self._index += 1

    def _scan_argument_list_if_present(self) -> None:
        """Advance over a ``(...)`` straight after the name, if there is one."""
        if self.is_done or self._expression[self._index] != "(":
            return
        self._index = _matching_paren(self._expression, self._index) + 1

    @staticmethod
    def _is_separator(character: str) -> bool:
        """Return whether ``character`` ends a name."""
        return character.isspace() or character in "&|()"


def _matching_paren(expression: str, opening: int) -> int:
    """Return the index of the ``)`` matching the ``(`` at ``opening``.

    >>> _matching_paren("a(b(c)d)e", 1)
    7
    """
    depth = 0
    for index in range(opening, len(expression)):
        depth += _nesting_delta(expression[index])
        if depth == 0:
            return index
    raise AssertionError(f"unclosed parenthesis in {expression!r}")


def _nesting_delta(character: str) -> int:
    """Return how ``character`` changes the parenthesis nesting depth."""
    if character == "(":
        return 1
    if character == ")":
        return -1
    return 0


def _parse_or(
    tokens: tuple[str, ...], position: int
) -> tuple[set[TestId], int]:
    """Parse a disjunction, the loosest-binding operator."""
    left, position = _parse_and(tokens, position)
    while position < len(tokens) and tokens[position] == "|":
        right, position = _parse_and(tokens, position + 1)
        left = left | right
    return left, position


def _parse_and(
    tokens: tuple[str, ...], position: int
) -> tuple[set[TestId], int]:
    """Parse a conjunction of possibly-negated terms."""
    left, position = _parse_term(tokens, position)
    while position < len(tokens) and tokens[position] == "&":
        right, position = _parse_term(tokens, position + 1)
        left = left & right
    return left, position


def _parse_term(
    tokens: tuple[str, ...], position: int
) -> tuple[set[TestId], int]:
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


def _predicate(token: str) -> set[TestId]:
    """Return the tests one leaf predicate selects."""
    if token == "kind(bench)":
        return set(BENCH_TESTS)
    if token == "kind(test)":
        return set(UNIVERSE) - BENCH_TESTS
    name, argument = _split_predicate(token)
    if name == "package":
        return _package_selection(argument)
    assert name == "test", f"model does not cover this predicate: {token!r}"
    return _test_name_selection(token, argument)


def _package_selection(package: str) -> set[TestId]:
    """Return the tests belonging to ``package``, rejecting unknown crates.

    An unmodelled crate must not read as an empty selection. ``not
    package(...)`` over an empty set removes nothing, so a filter naming a
    real crate the model holds no entry for would select the intended set *as
    modelled* while dropping that crate's tests in the workspace. Failing
    here is what makes such a filter visibly wrong instead.

    ``chutoro-providers-dense`` is a crate of the real workspace that the
    model holds no entry for, so asking for it raises rather than returning an
    empty set a ``not`` would then read as a no-op. The exception is raised
    rather than asserted so that pytest's assertion rewriting leaves its
    traceback alone and this doctest can pin the message.

    >>> sorted(_package_selection(DEV_PACKAGE))
    [('chutoro-core', 'bench::hnsw::throughput'), ('chutoro-core', 'gw::collide::case'), ('chutoro-core', 'hnsw::cpu::other::case'), ('chutoro-core', 'hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring')]
    >>> _package_selection("chutoro-providers-dense")
    Traceback (most recent call last):
    ...
    LookupError: model holds no test in package 'chutoro-providers-dense'
    """
    if not any(owner == package for owner, _ in UNIVERSE):
        raise LookupError(f"model holds no test in package {package!r}")
    return {(owner, name) for owner, name in UNIVERSE if owner == package}


def _test_name_selection(token: str, argument: str) -> set[TestId]:
    """Return the tests a ``test(=name)`` predicate selects.

    nextest 0.9.143 compares ``=`` against the whole registered test name,
    module path included, so this is an exact match and not a suffix match.
    Conflating the two would make every test whose name ends in the same
    segment identical.

    A name is not unique across packages: ``gw::collide::case`` is registered
    in two crates here, so this predicate returns both and only
    ``package(...)`` narrows it to one.

    >>> sorted(_test_name_selection("test(=hnsw::cpu::other::case)", "=hnsw::cpu::other::case"))
    [('chutoro-core', 'hnsw::cpu::other::case')]
    >>> sorted(_test_name_selection("test(=gw::collide::case)", "=gw::collide::case"))
    [('chutoro-core', 'gw::collide::case'), ('other-crate', 'gw::collide::case')]
    """
    assert argument.startswith("="), f"unsupported test predicate: {token!r}"
    named = argument.removeprefix("=")
    return {(owner, name) for owner, name in UNIVERSE if name == named}


def _split_predicate(token: str) -> tuple[str, str]:
    """Split ``name(argument)`` into its parts, rejecting any other shape."""
    assert token.endswith(")") and "(" in token, (
        f"model does not cover this predicate: {token!r}"
    )
    name, _, argument = token.partition("(")
    return name, argument[:-1]


def selection_of(expression: str) -> frozenset[TestId]:
    """Return the model selection, extending the universe to any unknown name.

    A test name outside `UNIVERSE` would otherwise select nothing at all, so
    ``not test(=something_else)`` would remove nothing and a filter naming an
    unrelated case would look identical to the intended one. The model
    therefore adds every name the expression mentions, under `_UNMODELLED_PACKAGE`.

    A ``package(...)`` argument gets no such extension, and cannot: ``not
    package(x)`` subtracts from the universe, so any fixed stand-in for an
    unmodelled crate is cancelled by the ``not``. `_package_selection` rejects
    such a crate instead, which a caller reading this selection sees as an
    error rather than as the intended set.

    >>> sorted(selection_of("not kind(bench) & not (package(chutoro-core) & test(=hnsw::cpu::other::case))"))
    [('chutoro-core', 'gw::collide::case'), ('chutoro-core', 'hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring'), ('other-crate', 'gw::collide::case'), ('other-crate', 'other-crate::test::case')]
    """
    return model_selection(expression) | _unknown_tests(expression)


def _unknown_tests(expression: str) -> frozenset[TestId]:
    """Return the mentioned test names the fixed universe does not hold.

    >>> sorted(_unknown_tests("test(=not_in_universe)"))
    [('unmodelled', 'not_in_universe')]
    """
    mentioned = frozenset(
        token[len("test(=") : -1]
        for token in _tokens(expression)
        if token.startswith("test(=") and token.endswith(")")
    )
    known = {name for _, name in UNIVERSE}
    return frozenset((_UNMODELLED_PACKAGE, name) for name in mentioned - known)


