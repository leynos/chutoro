"""Properties of the `cfg` evaluator that decides whether a proof is compiled.

The gate compiles without ``cfg(test)`` and with ``kani`` set, so a guard that
is false in that configuration removes its proof from the build. The evaluator
answers with three-valued logic: true, false, or unknown for configuration it
cannot know. Example tests pin the shapes that matter; these properties state
the algebra the evaluator must obey over generated predicates, where a rule
that only happens to work on the examples would be caught.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from kani_cfg_reading import _combine, _negate, _predicate, gated_by_test

#: One predicate leaf. Two are known under the gate's configuration and three
#: are configuration this evaluator cannot decide.
_LEAVES = st.sampled_from(
    ("test", "kani", 'feature = "opt"', 'target_os = "linux"', "unix")
)


def _predicates(depth: int = 0) -> st.SearchStrategy[str]:
    """Return a strategy for predicates nested up to `depth` combinators."""
    if depth >= 3:
        return _LEAVES
    inner = _predicates(depth + 1)
    combined = st.one_of(
        _LEAVES,
        inner.map(lambda body: f"not({body})"),
        st.tuples(inner, inner).map(lambda pair: f"all({pair[0]}, {pair[1]})"),
        st.tuples(inner, inner).map(lambda pair: f"any({pair[0]}, {pair[1]})"),
    )
    return combined


@given(_predicates())
def test_evaluation_is_deterministic(predicate: str) -> None:
    """The same predicate evaluates to the same verdict on every call."""
    assert _predicate(predicate) == _predicate(predicate)


@given(_predicates())
def test_double_negation_agrees_with_the_predicate(predicate: str) -> None:
    """`not(not(p))` is `p`, unknown included."""
    assert _predicate(f"not(not({predicate}))") == _predicate(predicate)


@given(_predicates())
def test_negation_is_three_valued(predicate: str) -> None:
    """Negation inverts a decided verdict and leaves an unknown unknown."""
    assert _predicate(f"not({predicate})") == _negate(_predicate(predicate))


@given(_predicates(), _predicates())
def test_all_and_any_follow_three_valued_logic(left: str, right: str) -> None:
    """A false conjunct decides `all`; a true disjunct decides `any`."""
    values = [_predicate(left), _predicate(right)]
    assert _combine("all", values) == _predicate(f"all({left}, {right})")
    assert _combine("any", values) == _predicate(f"any({left}, {right})")


@given(_LEAVES)
def test_an_unknown_leaf_never_excludes_a_proof(leaf: str) -> None:
    """Unknown configuration is not evidence that a proof is absent."""
    if _predicate(leaf) is None:
        assert not gated_by_test([f"#[cfg({leaf})]"])


@given(_predicates())
def test_a_guard_excludes_exactly_the_definitely_false_predicate(
    predicate: str,
) -> None:
    """The guard's answer is the verdict `false`, and nothing else.

    This is the property the gate depends on: a proof is treated as absent
    only when the predicate is definitely false under the gate's own
    configuration, never when it is true or undecidable.
    """
    assert gated_by_test([f"#[cfg({predicate})]"]) == (
        _predicate(predicate) is False
    )


@given(_predicates(), _predicates())
def test_a_conjunction_guard_is_exactly_all_of_its_conjuncts(
    left: str, right: str
) -> None:
    """`all` guards compose: the proof is absent iff one conjunct excludes it."""
    separate = gated_by_test([f"#[cfg({left})]", f"#[cfg({right})]"])
    combined = gated_by_test([f"#[cfg(all({left}, {right}))]"])
    assert combined == (
        _predicate(left) is False or _predicate(right) is False
    )
    assert combined == separate