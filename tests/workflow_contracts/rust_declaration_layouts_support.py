"""Generated Rust layouts for the declaration-reading property.

`rust_declaration_reading.declared_in_module` decides whether the nextest path
in `DEV_TEST_FILTER` names a case that exists. The table in the test module
names the shapes that must be refused; this module generates the sources that
table can only sample, composed from the same pieces in combinations no row
spells out.

The builders live here rather than beside the tests because together they state
the grammar of a generated source. A builder that emits a form Rust does not
accept makes the reader refuse the source for the wrong reason: the property
then passes without testing placement at all. Stating that rule once, beside
every builder that could break it, is what keeps the generated corpus honest.

Each placement is its own strategy. A tag string with a matching `if` chain
would let the tag and the built source drift apart, and the comment explaining
why a placement earns its verdict would sit away from the code that builds it.
"""

from __future__ import annotations

from hypothesis import strategies as st

#: The declaration the generated cases name. It matches the case the Makefile
#: exclusion excludes, so a generated verdict maps onto a real nextest path.
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


#: How deep the noise generator may nest modules inside modules.
_MAX_NESTING = 2

#: Filler comments. These are noise rather than a probe: a line comment's text
#: never follows the start of a line, and an inline block comment's does not
#: either, so the shared matcher reads no declaration out of them even when the
#: blanking is skipped. They still exercise the lexer on a source the reader
#: must handle.
_COMMENTS = (
    "    // a line comment\n",
    f"    // fn {_CASE}() {{}}\n",
    f"    /* fn {_CASE}() {{}} */\n",
)

#: Block comments with enough nesting that a reader stopping at the first `*/`
#: reads the tail as code. Every entry is brace-balanced at source level, so the
#: balance guard does not reject it, and each was checked to flip a verdict when
#: the nesting is ignored:
#:
#: - a brace in the inner comment, with its partner in the outer tail, shifts
#:   every later depth when the tail is miscounted as code;
#: - a declaration after the inner `*/` is read as a real one by a reader that
#:   stopped there, and the deeper placement requires the opposite verdict.
#:
#: These are the property's probe for nesting. Nothing else in the generator
#: can distinguish a nesting block reader from a first-`*/` one.
_NESTED_COMMENTS = (
    f"    /* outer /* inner */ fn {_CASE}() {{}} */\n",
    "    /* a /* b { */ } */\n",
    "    /* a /* b } */ { */\n",
)

#: Block comments whose *body line* is a complete declaration of the case.
#: That line starts at column zero, and `_FUNCTION` is anchored to the start of
#: a line, so it is the one shape the matcher reads out of a comment when the
#: blanking is skipped — which is what makes the property fail if `_blank` is
#: removed. Two body shapes are offered so the probe does not rest on one exact
#: spelling. Each keeps its braces balanced, so the balance guard stays
#: meaningful.
_COMMENTED_DECLARATIONS = (
    f"/*\nfn {_CASE}() {{}}\n*/\n",
    f"/*\nfn {_CASE}() {{\n}}\n*/\n",
)


def _noise_item(remaining: int, indent: int) -> st.SearchStrategy[str]:
    """Generate filler that cannot change the expected verdict.

    Filler declares only names other than `_CASE`, its modules are never called
    `tests`, and it emits comments. All three limits keep the expectation
    derivable from the one deliberate placement: a second top-level `tests`, or
    a second declaration of the case, would make the verdict depend on which
    reading the source happened to reach first.

    Filler comments here are noise, not a blanking probe. None of them can be
    read as a declaration: `_FUNCTION` is anchored to the start of a line, so a
    comment whose text opens the line — `// fn case()`, `/* fn case() */` — is
    invisible to it even without the blanking. The probe for that lives in
    `_COMMENTED_DECLARATIONS`, which puts the declaration on its own line
    inside a block comment.
    """
    declaration = st.builds(
        _declaration,
        indent=st.just(indent),
        qualifier=st.sampled_from(_QUALIFIERS),
        name=st.sampled_from(tuple(name for name in _NAMES if name != _CASE)),
    )
    if remaining == 0:
        return st.one_of(declaration, st.sampled_from(_COMMENTS))
    return st.one_of(
        declaration,
        st.sampled_from(_COMMENTS),
        st.builds(
            _module,
            st.lists(_noise_item(remaining - 1, indent + 4), max_size=2).map("".join),
            name=st.sampled_from(("nested", "deeper")),
            indent=st.just(indent),
        ),
    )


def _noise(indent: int = 4, nesting: int = _MAX_NESTING) -> st.SearchStrategy[str]:
    """Generate a run of filler items at the given indentation.

    ``nesting`` bounds how deeply filler may nest modules inside modules. It is
    a parameter rather than the module constant because one caller places the
    filler inside an `impl`, where a nested `mod` is not valid Rust: `impl`
    bodies accept associated items, and a module is not one. That caller passes
    0 so the filler still varies without ever opening a module.
    """
    return st.lists(_noise_item(nesting, indent), max_size=3).map("".join)


def _block(header: str, body: str, indent: int = 0) -> str:
    """Return any braced block — a module, an `impl`, or a plain `if`.

    Generated sources are assembled from this and `_module` rather than from
    literal fragments. A literal `"mod other {{"` in a non-f-string emits a
    doubled brace, and the resulting source is not Rust: the property then
    passes because the reader cannot read it at all, which proves nothing about
    declaration depth. Building through one pair of braces keeps every
    generated source well formed.
    """
    return f"{' ' * indent}{header} {{\n{body}{' ' * indent}}}\n"


@st.composite
def _own_level(draw: st.DrawFn) -> tuple[str, bool]:
    """Place the case at the top-level `tests` module's own level.

    This is the one shape nextest can select, so it is the only placement the
    property expects to pass.
    """
    declaration = _declaration(4, draw(st.sampled_from(_QUALIFIERS)))
    return draw(_noise(0)) + _module(f"{draw(_noise())}{declaration}"), True


@st.composite
def _deeper(draw: st.DrawFn) -> tuple[str, bool]:
    """Place the case one layer deeper inside `mod tests`.

    A case moved into a nested module has a longer path than the filter names,
    and nextest selects by the trailing segment, so this placement is a
    missing exclusion and must be refused.
    """
    declaration = _declaration(4, draw(st.sampled_from(_QUALIFIERS)))
    noise = draw(_noise())
    # Both openers are items, so a module may legally contain either. `if` is a
    # statement and cannot sit at module level: generating one there would make
    # the source invalid Rust, and an invalid source is refused for the wrong
    # reason.
    #
    # The filler's nesting limit follows the opener for the same reason. A
    # module accepts a nested module; an `impl` accepts only associated items,
    # so a `mod` inside `impl Holder` is not Rust either. Filler that ignores
    # this reintroduces exactly the invalid-source class the choice of openers
    # exists to exclude.
    opened = draw(st.sampled_from(("mod other", "impl Holder")))
    filler_nesting = 0 if opened == "impl Holder" else _MAX_NESTING
    body = f"{draw(_noise(8, filler_nesting))}    {declaration}"
    nested = _block(opened, body, indent=4)
    # A commented declaration at the module's *own* level is what makes the
    # property catch a reader that stopped blanking comments. On its own,
    # comment filler proves nothing: a commented declaration nested beside the
    # real one is rejected by depth either way. At depth 1 it is decisive — an
    # unblanked reader finds it, answers True, and fails this False expectation.
    #
    # The nested forms go here for the same reason: a reader that ends a block
    # comment at the first `*/` reads the outer tail as code, and in this
    # placement that tail makes it answer True. Only comment layouts whose
    # evidence is a *refused* verdict belong here, because this is the one
    # placement that expects False.
    #
    # One nested form is emitted unconditionally rather than drawn. Drawn, it
    # lands in the one-in-four deeper placement and then one-in-five within it,
    # so a seed could explore a hundred examples without ever reaching it —
    # measured at five seeds in eight against a reader that stopped at the
    # first `*/`. Emitting it always makes the property's nesting evidence
    # independent of the seed. The rest are drawn, so the generator still varies
    # their neighbours.
    always, *sometimes = _NESTED_COMMENTS
    drawn = st.sampled_from(_COMMENTED_DECLARATIONS + tuple(sometimes))
    commented = always + draw(drawn)
    return _module(f"{noise}{commented}{nested}"), False


@st.composite
def _sibling(draw: st.DrawFn) -> tuple[str, bool]:
    """Place the case in a top-level module that is not `tests`.

    This is the placement that separates the reader's two conditions from each
    other. A declaration in a sibling top-level module sits at exactly the
    depth one inside `mod tests` does, so the depth test alone accepts it; only
    the span refuses it. Without this placement, deleting the span condition
    leaves every generated example passing.
    """
    declaration = _declaration(4, draw(st.sampled_from(_QUALIFIERS)))
    noise = draw(_noise())
    tests = _module(f"{noise}    fn other() {{}}\n")
    sibling = _module(f"{noise}{declaration}", name="other")
    before, after = draw(st.sampled_from(((tests, sibling), (sibling, tests))))
    return before + after, False


@st.composite
def _nested_tests(draw: st.DrawFn) -> tuple[str, bool]:
    """Wrap the whole `tests` module inside another module.

    This placement is what separates a reader that requires `tests` at the file
    root from one that only requires the declaration at *some* module's own
    level. Without it, a mutant that ignores the module's depth passes every
    generated example.
    """
    declaration = _declaration(4, draw(st.sampled_from(_QUALIFIERS)))
    noise = draw(_noise())
    inner = _module(f"{draw(_noise(8))}    {declaration}", indent=4)
    return _block("mod outer", f"{noise}{inner}"), False


#: The four placements the property covers, and the verdict each implies. The
#: expectation comes from how the source was built, not from the reader under
#: test: asking the reader for the expectation would reduce the property to
#: `assert f(x) == f(x)`.
LAYOUTS: st.SearchStrategy[tuple[str, bool]] = st.one_of(
    _own_level(), _deeper(), _sibling(), _nested_tests()
)
