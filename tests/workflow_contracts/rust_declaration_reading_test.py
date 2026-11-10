"""Tests for the module-path reading behind the poisoned-lock exclusion.

`rust_declaration_reading.declared_in_module` decides whether the nextest path
in `DEV_TEST_FILTER` names a case that exists. Two kinds of test state that
here: a table of the layouts the path must refuse, and a generated one for the
relation the table can only sample. The table says which named shapes are
wrong; the property says that no composition of those pieces reaches a verdict
the generator did not intend, which is what separates a reader that tracks
nesting from one that happens to match today's file.

The builders and the four placement strategies live in
`rust_declaration_layouts_support`. They state the grammar of a generated
source, and a builder that emits a form Rust does not accept would make the
reader refuse the source for the wrong reason. Keeping that rule in one module,
beside every builder that could break it, is what keeps the corpus honest.
"""

from __future__ import annotations

import typing as typ

import pytest
from hypothesis import given

from rust_declaration_layouts_support import LAYOUTS, _declaration, _module
from rust_declaration_reading import declared_in_module

_CASE = "graph_helpers_report_poisoning_without_reconfiguring"


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
        # A reader that ended a block comment at the first `*/` would read the
        # tail as code. The declaration after the inner `*/` is then a real one
        # at the module's own level, so the row flips to True. The property
        # covers this too, but through a generated layout; this row states the
        # same fact in one shape nothing has to sample.
        pytest.param(
            _module(f"    /* outer /* inner */ fn {_CASE}() {{}} */\n"),
            False,
            id="declaration-in-a-nested-comment",
        ),
        # The brace pair straddles the inner boundary. Without nesting the tail
        # is code, which both shifts the depth of the declaration below and
        # leaves the source unbalanced.
        pytest.param(
            _module(f"    /* a /* b {{ */ }} */\n{_declaration(4)}"),
            True,
            id="braces-across-a-nested-comment-boundary",
        ),
        pytest.param(
            _module(f"    /* a /* b }} */ {{ */\n{_declaration(4)}"),
            True,
            id="reverse-braces-across-a-nested-comment-boundary",
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



@given(layout=LAYOUTS)
def test_generated_layouts_agree_with_where_the_declaration_was_placed(
    layout: tuple[str, bool],
) -> None:
    """Accept exactly the generated sources that declare the case in `mod tests`.

    The table names the layouts a reader must distinguish; this states the
    relation behind them over sources composed from the same pieces in
    combinations no row spells out — several modules, repeated names,
    declarations qualified several ways, and comment filler.

    The balance guard comes first deliberately. A generator bug that emits a
    stray brace makes every source unreadable, and an unreadable source is
    refused — so the property would pass without ever testing the relation it
    claims to. That happened once here, with doubled braces from a literal
    fragment; the guard turns that class of bug into a failure instead of a
    green run.

    It is a guard, not a grammar check. Braces can balance in a source Rust
    still rejects, such as a `mod` inside an `impl` body, and such a source is
    refused for the wrong reason just as surely. That class is kept out where
    it belongs: the support module's builders bound what they emit, and its
    comments state why the bound exists.
    """
    source, expected = layout
    assert source.count("{") == source.count("}"), (
        f"generated source is not balanced Rust:\n{source}"
    )
    actual = declared_in_module(source, "tests", _CASE)
    assert actual is expected, (
        f"expected {expected} for this layout, got {actual}:\n{source}"
    )
