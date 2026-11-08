"""Tests for the module-path reading behind the poisoned-lock exclusion.

`rust_declaration_reading.declared_in_module` decides whether the nextest path
in `DEV_TEST_FILTER` names a case that exists. Two kinds of test state that
here: a table of the layouts the path must refuse, and a generated one for the
relation the table can only sample. The table says which named shapes are
wrong; the property says that no composition of those pieces reaches a verdict
the generator did not intend, which is what separates a reader that tracks
nesting from one that happens to match today's file.
"""

from __future__ import annotations

import typing as typ

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rust_declaration_reading import declared_in_module

_CASE = "graph_helpers_report_poisoning_without_reconfiguring"

#: The declarations the generator may emit, as (qualifier, name). A qualifier
#: the shared matcher reads loosely, such as a generic list, is deliberately
#: absent: the generator states placement, and the matcher's own coverage of
#: signatures belongs with the reading that owns it.
_NAMES = (_CASE, "other", f"not_{_CASE}")
_QUALIFIERS = ("", "pub ", "async ", "unsafe ", "pub async ")


def _declaration(indent: int = 0, qualifier: str = "", name: str = _CASE) -> str:
    """Return one declaration line, indented and qualified as asked."""
    return f"{' ' * indent}{qualifier}fn {name}() {{}}\n"


def _module(body: str, *, name: str = "tests", indent: int = 0) -> str:
    """Return a module, indented if it is going inside another one."""
    return f"{' ' * indent}mod {name} {{\n{body}{' ' * indent}}}\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        pytest.param(_module(_declaration(4)), True, id="directly-inside-mod-tests"),
        pytest.param(
            _module(f"    mod nested {{\n        fn {_CASE}() {{}}\n    }}\n"),
            False,
            id="inside-a-nested-module",
        ),
        pytest.param(
            _module(
                f"    mod nested {{\n        mod deeper {{\n"
                f"            fn {_CASE}() {{}}\n        }}\n    }}\n"
            ),
            False,
            id="inside-a-doubly-nested-module",
        ),
        pytest.param(
            _module(f"    if true {{\n        fn {_CASE}() {{}}\n    }}\n"),
            False,
            id="inside-a-block",
        ),
        pytest.param(
            _module(f"    impl Holder {{\n        fn {_CASE}() {{}}\n    }}\n"),
            False,
            id="inside-an-impl",
        ),
        pytest.param(
            _module(f"    // fn {_CASE}() {{}}\n"),
            False,
            id="inside-a-line-comment",
        ),
        pytest.param(
            _module(f"    /*\n    fn {_CASE}() {{}}\n    */\n"),
            False,
            id="inside-a-block-comment",
        ),
        pytest.param(_declaration(), False, id="outside-any-module"),
        pytest.param(
            _declaration() + _module("    fn other() {}\n"),
            False,
            id="before-mod-tests-opens",
        ),
        pytest.param(
            _module("    fn other() {}\n") + _declaration(),
            False,
            id="after-mod-tests-closes",
        ),
        pytest.param(
            _module(_declaration(4), name="other"), False, id="a-different-module"
        ),
        pytest.param(
            _module(_module(_declaration(8), indent=4), name="other"),
            False,
            id="tests-nested-inside-another-module",
        ),
        pytest.param(
            _module(_declaration(4, name=f"not_{_CASE}")),
            False,
            id="longer-name-suffix",
        ),
        pytest.param(
            _module(_declaration(4, qualifier="pub ")),
            True,
            id="public-declaration",
        ),
        pytest.param(
            _module(_declaration(4, qualifier="async ")),
            True,
            id="async-declaration",
        ),
        pytest.param(
            _module(f'    fn helper() {{ let brace = "}}"; }}\n{_declaration(4)}'),
            True,
            id="brace-in-a-string-before-the-declaration",
        ),
        pytest.param(
            _module(f"    fn helper() {{ let brace = '}}'; }}\n{_declaration(4)}"),
            True,
            id="brace-in-a-char-literal-before-the-declaration",
        ),
        pytest.param(
            _module(_declaration(8)),
            True,
            id="directly-inside-an-indented-module",
        ),
        pytest.param(
            _module(_declaration(4), name="tests") + _module(_declaration(4)),
            True,
            id="declared-in-both-tests-modules",
        ),
        # A sibling top-level module is the one layout the depth test alone
        # accepts: its declarations sit at the same brace depth as `mod tests`
        # ones, so only the span separates them. Without this row, deleting the
        # span condition leaves every other row passing. Both orderings are
        # covered, because a reader that anchored on the first match could
        # otherwise pass by luck.
        pytest.param(
            _module("    fn other() {}\n", name="tests")
            + _module(_declaration(4), name="other"),
            False,
            id="at-module-depth-in-a-later-sibling",
        ),
        pytest.param(
            _module(_declaration(4), name="other")
            + _module("    fn other() {}\n", name="tests"),
            False,
            id="at-module-depth-in-an-earlier-sibling",
        ),
        # The trailing block comment is load-bearing. Without a later `*/` the
        # false block opened by the `/*` inside the line comment has nothing to
        # run to, and the row passes even with the wrong stripping order.
        pytest.param(
            _module(f"    // glob tests/*.rs\n{_declaration(4)}") + "/* trailing */\n",
            True,
            id="block-opener-inside-a-line-comment",
        ),
        pytest.param(
            _module(f"    /* see // below\n{_declaration(4)}    */\n"),
            False,
            id="line-comment-inside-a-block-comment",
        ),
    ),
)
def test_declaration_must_sit_at_the_named_modules_own_level(
    source: str, expected: bool
) -> None:
    """Every layout that is not the named case in the named module is refused.

    The false rows are the point: each is a source a bare `fn {case}(` probe
    accepted, and each leaves the Makefile exclusion naming nothing while
    `dev-test` goes on running the Cranelift-aborting case.
    """
    actual = declared_in_module(source, "tests", _CASE)
    assert actual is expected, (
        f"expected {expected} for this layout, got {actual}:\n{source}"
    )


#: How deep the noise generator may nest modules inside modules.
_MAX_NESTING = 2


def _noise_item(remaining: int, indent: int) -> st.SearchStrategy[str]:
    """Generate filler that cannot change the expected verdict.

    Filler declares only names other than `_CASE`, and its modules are never
    called `tests`. Both limits are what keep the expectation derivable from
    the one deliberate placement: filler that could open a second top-level
    `tests`, or declare the case itself, would make the expected verdict
    depend on which of two readings the source happened to reach first.
    """
    declaration = st.builds(
        _declaration,
        indent=st.just(indent),
        qualifier=st.sampled_from(_QUALIFIERS),
        name=st.sampled_from(tuple(name for name in _NAMES if name != _CASE)),
    )
    if remaining == 0:
        return declaration
    return st.one_of(
        declaration,
        st.builds(
            _module,
            st.lists(_noise_item(remaining - 1, indent + 4), max_size=2).map("".join),
            name=st.sampled_from(("nested", "deeper")),
            indent=st.just(indent),
        ),
    )


def _noise(indent: int = 4) -> st.SearchStrategy[str]:
    """Generate a run of filler items at the given indentation."""
    return st.lists(_noise_item(_MAX_NESTING, indent), max_size=3).map("".join)


#: How a generated source places `mod tests` relative to the declaration.
_PLACEMENTS = (
    "at-the-modules-own-level",
    "deeper-in-the-module",
    "inside-another-module",
)


@st.composite
def _layouts(draw: st.DrawFn) -> tuple[str, bool]:
    """Generate a source and the verdict its shape earns.

    The expected verdict comes from how the source was built, not from the
    reader under test. Exactly one of three placements is drawn: the case at
    the top-level `tests` module's own level, the case one layer deeper inside
    it, or a `tests` module that is itself nested inside another module. Only
    the first is a path nextest can select, and only the first is expected to
    pass. Asking the reader for the expectation would reduce the property to
    `assert f(x) == f(x)`.

    The third placement is what separates a reader that requires `tests` at the
    file root from one that only requires the declaration at *some* module's own
    level. Without it, a mutant that ignores the module's depth passes every
    generated example.
    """
    placement = draw(st.sampled_from(_PLACEMENTS))
    declaration = _declaration(4, draw(st.sampled_from(_QUALIFIERS)))
    noise = draw(_noise())
    # A module named `tests` is only ever the one written here. Filler modules
    # carry other names, so the verdict cannot depend on which of two same-named
    # modules the reader reached first.
    if placement == "at-the-modules-own-level":
        return f"{draw(_noise(0))}mod tests {{\n{noise}{declaration}}}\n", True
    if placement == "deeper-in-the-module":
        opened = draw(
            st.sampled_from(("mod other {{\n", "impl Holder {{\n", "if true {{\n"))
        )
        body = f"{noise}    {opened}{draw(_noise(8))}    {declaration}    }}\n"
        return f"mod tests {{\n{body}}}\n", False
    body = f"{noise}    mod tests {{\n{draw(_noise(8))}    {declaration}    }}\n"
    return f"mod outer {{\n{body}}}\n", False


@given(layout=_layouts())
def test_generated_layouts_agree_with_where_the_declaration_was_placed(
    layout: tuple[str, bool],
) -> None:
    """Accept exactly the generated sources that declare the case in `mod tests`.

    The table names the layouts a reader must distinguish; this states the
    relation behind them over sources composed from the same pieces in
    combinations no row spells out — several modules, repeated names, and
    declarations qualified several ways.
    """
    source, expected = layout
    actual = declared_in_module(source, "tests", _CASE)
    assert actual is expected, (
        f"expected {expected} for this layout, got {actual}:\n{source}"
    )
