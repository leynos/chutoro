"""Read a Rust source for the tests that pay a nested build's cost.

`trybuild_override_test` states the policy; this module supplies the reading it
states the policy in. They are separated because the reading has its own cases,
a Rust source having comments, four kinds of literal and a lifetime marker that
looks like a fifth, and because the policy module reached the 400-line limit
`AGENTS.md` sets once those cases arrived.

Nothing here asserts anything. A reader that fails closed returns nothing and
leaves the caller to decide whether that is an offence.

It is not a Rust parser and does not try to be. It knows where code stops and a
comment or a literal begins, which is all that is needed to count braces and to
tell a call from a mention. Everything it cannot read reads as code, so the
failure direction is over-reporting, which the pinned set in the policy module
catches.
"""

import collections.abc as cabc
import pathlib
import re
import typing as typ

from rust_scanning import (
    _Region,
    _blank,
    _brace_depths,
    _literal_spans,
    _matching_brace,
    _outside,
    _regions,
    _without_comments,
)

from workflow_support import ROOT as REPO_ROOT

#: Sources of the cost this contract is about. A constructed ``TestCases``
#: compiles a scratch crate; a spawned ``cargo`` builds a fixture workspace.
#: Naming the crate is not enough for either: an import that is never used, or
#: a paragraph explaining why a harness was removed, costs nothing.
COST_MARKERS: typ.Final[tuple[str, ...]] = (
    "TestCases::new(",
    'Command::new(env!("CARGO"))',
    'Command::new("cargo")',
)

#: Whitespace Rust permits around a path separator or before a call's
#: parenthesis, which the formatter removes. Normalised rather than depended
#: upon, so the reading does not rest on the formatter having run.
_SPACED_SYNTAX = re.compile(r"\s*(::)\s*|\s+(\()")

#: A test function's declaration. `#[test]` and its cfg attributes may sit
#: between the attribute and the signature, so the name is taken from the
#: signature and the attribute is found by walking backwards.
#:
#: Every shape a test may take, because the narrow ones fail in the
#: under-reporting direction this module exists to close. `async fn` is the
#: only way a `#[tokio::test]` or `#[async_std::test]` test is written, and
#: both attributes are named below, so a matcher without `async` disagreed
#: with the attribute set it is paired with. A `fn` inside `mod tests { ... }`
#: is indented. A name may carry an uppercase character. A test in any of
#: those shapes was discovered by nothing, and a pinned set records only the
#: tests the tree holds today, so the missing override passed.
#:
#: The trailing `\(` stays: the body scan starts from the match's end.
_FUNCTION = re.compile(
    r"^[ \t]*"
    r"(?:pub(?:\s*\([^)]*\))?[ \t]+)?"
    r"(?:async[ \t]+)?"
    r"(?:unsafe[ \t]+)?"
    r"(?:extern[ \t]+\"[^\"]*\"[ \t]+)?"
    r"fn[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    # An `#[rstest]` function may be generic or carry a lifetime, and the list
    # nests (`<T: Into<String>>`), so it is consumed up to the last `>` before
    # the parameter list rather than by a bracket-counting pattern.
    r"(?:\s*<[^{;]*?>)?\s*\(",
    re.MULTILINE,
)


def _normalised(text: str) -> str:
    """Return the text with spacing around `::` and `(` removed.

    Rust permits the spacing and this repository's formatter removes it, so the
    reading normalises rather than depending on the formatter having run.

    Applied to a body once its extent is known, never to the whole source:
    collapsing spacing changes the text's length, and every index used to find
    that extent is the source's own.

    Returns
    -------
    str
        The text with that spacing collapsed.
    """
    return _SPACED_SYNTAX.sub(lambda found: found.group(1) or found.group(2), text)


def _body_span(
    text: str, literals: list[tuple[int, int]], start: int
) -> tuple[int, int] | None:
    """Return one function body's extent, by matching braces from its signature.

    Braces inside a literal are skipped, and the text is already comment-free,
    so only a brace in code opens or closes the body. Counting one from a
    literal would end the body early and hide every marker after it, and the
    pinned set would then read as a repository with one fewer costly test
    rather than failing.

    Positions are the source's own throughout, because blanking preserves
    them; the text is not re-indexed at any point.

    Returns
    -------
    tuple[int, int] or None
        The half-open extent of the body, or None when no brace in code
        follows the signature. Unbalanced braces yield the rest of the source,
        which reads as a larger body and so can only over-report.
    """
    opening = next(
        (
            index
            for index in range(start, len(text))
            if text[index] == "{" and _outside(index, literals)
        ),
        None,
    )
    if opening is None:
        return None
    depth = 0
    for index in range(opening, len(text)):
        if not _outside(index, literals):
            continue
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return opening, index + 1
    return opening, len(text)


def function_body(text: str, name: str) -> str | None:
    """Return one named function's body as code, or None when it is absent.

    The extent runs from the opening brace through its matching brace, with
    comments and literals blanked and spacing around `::` and `(` collapsed.
    Blanking both is what makes this reading answer where a *call* is rather
    than where a name is written: a marker in a comment or a string literal is
    a mention, and a mention is not a dispatch. Blanking preserves positions,
    so the extent the brace scan finds is still the source's own.

    A source holding several functions is read function by function, so a
    marker in a sibling function is not attributed to this one. Names are
    compared exactly; a call, a mention in a comment and a declaration in a
    nested module are all a different name.

    Parameters
    ----------
    text
        The contents of one Rust source file.
    name
        The function whose body is wanted, without any qualifier.

    Examples
    --------
    >>> function_body("fn a() { work(); }", "a")
    '{ work(); }'
    >>> function_body("fn b() { work(); }", "a") is None
    True
    >>> 'work' in function_body("fn a() { // work\\n run(); }", "a")
    False
    >>> 'run' in function_body("fn a() { // work\\n run(); }", "a")
    True
    >>> 'into_par_iter' in function_body(
    ...     'fn a() { let note = "into_par_iter"; run(); }', "a"
    ... )
    False

    Returns
    -------
    str or None
        The body as code, or None when no function of that name is declared.
    """
    regions = _regions(text)
    # The declaration is found on the comment-blanked source, whose positions
    # are the text's own; the extent is then taken from the fully blanked
    # source so a brace or marker inside a literal cannot end the body early.
    comments = (region for region in regions if region.is_comment)
    declarations = _blank(text, comments)
    code = _blank(text, regions)
    literals = _literal_spans(regions)
    for match in _FUNCTION.finditer(declarations):
        if match["name"] != name:
            continue
        span = _body_span(declarations, literals, match.end())
        if span is None:
            return None
        return _normalised(code[span[0] : span[1]]).strip()
    return None


#: Attributes that make a function a test nextest will run and therefore
#: bound. `#[test]` alone is not enough: this repository declares tests with
#: `rstest` as a matter of course, and a costly test written that way would be
#: discovered by nothing while the pinned set stayed unchanged.
#:
#: Matched on the attribute's path rather than the whole attribute, so
#: `#[rstest]`, `#[rstest(...)]` and `#[tokio::test]` are all recognised and
#: `#[test_only_helper]` is not.
_TEST_ATTRIBUTES: typ.Final[frozenset[str]] = frozenset({
    "test",
    "rstest",
    "tokio::test",
    "async_std::test",
})

_ATTRIBUTE_PATH = re.compile(r"^#\[\s*(?P<path>[A-Za-z_][A-Za-z0-9_:]*)")


def _attributes_before(text: str) -> list[str]:
    """Return the lines before a signature, innermost first, attributes joined.

    An attribute may span lines. `#[tokio::test(\n    flavor = "multi_thread"\n)]`
    and an `#[case(...)]` with a long value are both ordinary Rust, and a
    line-wise walk meets `)]` first: it is not an attribute opening, so the walk
    stopped there and the test above it was never found. That under-reports,
    which is the direction nothing else catches, because the test simply drops
    out of discovery rather than failing anything.

    A continuation is joined to the `#[` that opened it. The join is driven by
    that opening rather than by counting brackets, because the caller keeps
    string literals intact and a bracket inside one would throw a count off.
    """
    joined: list[str] = []
    pending: list[str] = []
    for line in reversed(text.rstrip().split("\n")):
        stripped = line.strip()
        if pending:
            pending.insert(0, stripped)
            if stripped.startswith("#["):
                joined.append(" ".join(pending))
                pending = []
            continue
        if stripped.endswith("]") and not stripped.startswith("#["):
            pending = [stripped]
            continue
        joined.append(stripped)
    # An unterminated run is not an attribute; its lines are returned as they
    # were so the walk sees code and stops, rather than silently swallowing
    # everything above the signature.
    joined.extend(pending)
    return joined


def _is_test(text: str, start: int) -> bool:
    """Return whether the declaration at `start` is attributed as a test.

    Attributes are read backwards from the signature, because `#[cfg(...)]`
    and doc comments may sit between the two.

    Returns
    -------
    bool
        True when one of the preceding attributes names a test attribute.
    """
    for line in _attributes_before(text[:start]):
        stripped = line
        # The caller walks the comment-blanked source, so a comment between the
        # attribute and the signature arrives here as a line of spaces. Nothing
        # tests for a comment prefix, because none survives the blanking; the
        # walk steps over the blank instead. A walk that stopped at it found no
        # attribute at all, and the test it was reading paid a nested build's
        # cost with nothing naming an allowance for it.
        if not stripped:
            continue
        # Rust permits spacing around `::`, so `#[tokio :: test]` is the same
        # attribute as `#[tokio::test]`. Normalising first keeps the attribute
        # set and the recognition of it from disagreeing over whitespace.
        found = _ATTRIBUTE_PATH.match(_normalised(stripped))
        if found is not None and found["path"] in _TEST_ATTRIBUTES:
            return True
        if not stripped.startswith("#["):
            return False
    return False


def compile_contract_tests(text: str) -> list[str]:
    """Return every test in the source that pays a nested build's cost.

    A test pays it by constructing a `trybuild::TestCases` harness, which
    compiles a scratch crate against this workspace's dependency graph, or by
    spawning a nested `cargo`, which builds a fixture workspace. Both cost what
    a cold build costs and both overrun the base allowance.

    The reading is per test function rather than per file, because the defect
    this contract exists to catch is a sibling: one test of a pair named in an
    override and the other left on the base allowance.

    Comments are excluded and literals are not. A marker in a comment is a
    mention, and this repository has a module whose documentation explains at
    length why a trybuild harness was removed; a marker inside a literal is
    part of a call, because one of the markers contains a string.

    Parameters
    ----------
    text
        The contents of one Rust source file.

    Examples
    --------
    >>> compile_contract_tests(
    ...     "#[test]\\nfn a() { let c = trybuild::TestCases::new(); }"
    ... )
    ['a']
    >>> compile_contract_tests("#[test]\\nfn b() { assert!(true); }")
    []
    >>> compile_contract_tests("fn c() { let c = TestCases::new(); }")
    []
    >>> compile_contract_tests(
    ...     "#[rstest]\\nfn d() { let c = trybuild::TestCases::new(); }"
    ... )
    ['d']

    Returns
    -------
    list[str]
        The name of each test whose body constructs a trybuild harness or
        spawns a nested cargo, in declaration order.
    """
    regions = _regions(text)
    # One text, with comments blanked and literals kept, so every index below
    # is the source's own. Normalising the whole source first would change its
    # length and the spans would no longer line up.
    source = _blank(text, (region for region in regions if region.is_comment))
    literals = _literal_spans(regions)
    found = []
    for match in _FUNCTION.finditer(source):
        if not _is_test(source, match.start()):
            continue
        span = _body_span(source, literals, match.end())
        if span is None:
            continue
        body = _normalised(source[span[0] : span[1]])
        if any(marker in body for marker in COST_MARKERS):
            found.append(match["name"])
    return found


#: Directories the sweep never enters: build output and the repository-local uv
#: cache and tool directories, which hold other repositories' checkouts.
IGNORED_DIRECTORIES = frozenset({"target", ".uv-cache", ".uv-tools"})


def discovered_tests(root: pathlib.Path = REPO_ROOT) -> dict[str, str]:
    """Return each compile-contract test, mapped to the file declaring it.

    Parameters
    ----------
    root : pathlib.Path
        The tree to read. Defaults to this repository, which is what every
        caller here wants; it is a parameter so the sweep can be driven over a
        tree built for the purpose. Parameterized over this repository's own
        sources the sweep agrees with itself whatever it does, because the
        answer it gives is the answer the assertion is written from.

    Returns
    -------
    dict[str, str]
        Test name to the path of its source file, relative to `root`.

    Notes
    -----
    An unreadable or undecodable source raises rather than being skipped. A
    sweep that swallowed the error would report a smaller set, and a smaller
    set is exactly how a costly test goes missing from the override list
    without anything failing. Raising names the file.
    """
    return {
        name: path.relative_to(root).as_posix()
        for path in sorted(root.rglob("tests/**/*.rs"))
        # Relative to `root`, because a checkout that itself sits under a
        # directory named `target` would otherwise lose every file. The uv
        # directories hold checkouts of other repositories (the CV-005
        # contracts run from one), whose tests are not this workspace's.
        if not IGNORED_DIRECTORIES.intersection(path.relative_to(root).parts)
        for name in compile_contract_tests(path.read_text(encoding="utf-8"))
    }
