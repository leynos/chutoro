"""Discover the tests that pay a nested build's cost.

A ``trybuild`` harness or a spawned nested ``cargo`` costs what a cold build
costs, so those tests need an allowance of their own. Finding them is a
per-test reading rather than a per-file one: the defect this contract exists to
catch is a sibling left on the base allowance, not a whole file that missed the
list. The sweep also maps each test to its file, which is what the override
contract compares.

Re-use policy: import from :mod:`trybuild_override_test` only.
"""

from __future__ import annotations

import pathlib
import re
import typing as typ

from rust_scanning import _blank, _literal_spans, _regions
from rust_source_reading import (
    COST_MARKERS,
    _ATTRIBUTE_PATH,
    _FUNCTION,
    _TEST_ATTRIBUTES,
    _body_span,
    _normalised,
)
from workflow_support import ROOT as REPO_ROOT


def _is_attribute(stripped: str) -> bool:
    """Report whether a line is an attribute's opening line.

    Rust writes `#` and `[` as separate tokens, so `# [tokio::test]` opens an
    attribute exactly as `#[tokio::test]` does. Testing for the adjacent pair
    would stop the walk at a spaced attribute and lose the test it decorates.
    """
    return stripped.startswith("#[") or stripped.startswith("# [")


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
            if _is_attribute(stripped):
                joined.append(" ".join(pending))
                pending = []
            continue
        if stripped.endswith("]") and not _is_attribute(stripped):
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
        if not _is_attribute(stripped):
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
