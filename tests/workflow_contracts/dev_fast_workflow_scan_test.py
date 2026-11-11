"""Contract tests for the scanner the dev-fast workflow checks share.

The CI contracts need to read arbitrary YAML: a job step is a mapping of
scalars, lists, and nested mappings, and Cargo commands hide in its strings.
`string_values` is the traversal those checks rely on, so its invariant is a
property over generated YAML shapes rather than a handful of examples. It is
pinned against an independent depth-first walk, because a scan that walked its
input with the helper under test would agree with it by construction.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from hypothesis import given
from hypothesis import strategies as st


def string_values(value: Any) -> Iterator[str]:
    """Yield workflow scalar strings recursively for focused config checks.

    Containers are visited in their natural order and non-string scalars are
    skipped, so a reading of one step cannot silently reorder or invent values.

    Parameters
    ----------
    value : Any
        A YAML-shaped value: a string, a mapping, a list, or any other scalar.

    Yields
    ------
    str
        Each string found, in depth-first order.

    Examples
    --------
    >>> list(string_values({"a": ["b", 1], "c": None}))
    ['b']
    >>> list(string_values([1, ["x", {"y": "z"}]]))
    ['x', 'z']
    """
    match value:
        case str():
            yield value
        case dict():
            for nested in value.values():
                yield from string_values(nested)
        case list():
            for nested in value:
                yield from string_values(nested)


def test_string_values_preserves_container_order_and_omits_non_strings() -> None:
    """The workflow scanner must flatten order-preservingly, skipping non-strings.

    The scanner decides which Cargo commands the fragment rules inspect, so it
    has to reach every string while reporting the ones it does not traverse.
    """
    workflow_value = {
        "first": "one",
        "nested": [
            "two",
            {"third": "three", "ignored": 7, "fourth": ["four", None]},
            False,
        ],
        "also_ignored": None,
    }

    assert list(string_values(workflow_value)) == ["one", "two", "three", "four"], (
        "workflow value traversal must keep container order and drop non-strings"
    )


#: YAML-like scalars that are not strings. The traversal must ignore every one
#: of them, including the ones Python would happily print.
_NON_STRING_SCALARS = st.one_of(
    st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False)
)


@st.composite
def _yaml_like_values(draw: st.DrawFn) -> Any:
    """Build one arbitrary YAML-shaped value for the traversal property.

    YAML mappings and sequences nest inside each other, which is exactly the
    recursion `string_values` implements, so the generator descends instead of
    emitting a fixed depth.
    """
    return draw(
        st.recursive(
            st.one_of(st.text(), _NON_STRING_SCALARS),
            lambda children: st.one_of(
                st.lists(children), st.dictionaries(st.text(), children)
            ),
            max_leaves=25,
        )
    )


def _collect_strings(value: Any) -> list[str]:
    """Collect the strings of a YAML-shaped value in depth-first order.

    Deliberately a second, differently shaped implementation. A property test
    that walked its input with the helper under test would agree with it by
    construction and prove nothing; this one states the order the contract
    promises in its own terms. The worklist keeps that independence while
    still reading the shape structurally, so the two implementations differ
    in traversal as well as in recursion.

    >>> _collect_strings({"a": ["b", 1], "c": None})
    ['b']
    >>> _collect_strings([1, ["x", {"y": "z"}]])
    ['x', 'z']
    """
    found: list[str] = []
    pending = [value]
    while pending:
        match pending.pop():
            case str() as text:
                found.append(text)
            case dict() as mapping:
                pending.extend(reversed(list(mapping.values())))
            case list() as items:
                pending.extend(reversed(items))
    return found


@given(value=_yaml_like_values())
def test_string_values_emits_exactly_the_nested_strings_in_depth_first_order(
    value: Any,
) -> None:
    """The traversal is an order-preserving, non-string-dropping projection.

    CodeRabbit asked for this as a property rather than more examples: the
    helper recurses over arbitrary YAML, so the invariant has to hold for
    nesting shapes nobody wrote a case for. Depth-first order is what makes
    the resulting scan readable as the workflow source.
    """
    assert list(string_values(value)) == _collect_strings(value), (
        f"traversal of {value!r} must match an independent depth-first walk"
    )
