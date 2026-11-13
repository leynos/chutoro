"""Read which module a Rust function is declared in.

The development route excludes one poisoned-lock regression by nextest path:
``hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring``.
nextest selects a case by that trailing module segment and name, and an
exclusion that matches nothing is not an error, so a declaration that moved
into a nested module would leave the aborting case running under Cranelift with
nothing reporting it.

Only the shape that path can mean is accepted: ``case`` declared *directly*
inside a top-level ``mod tests``. A declaration in a nested module or an
``impl`` has a longer path, one inside a comment does not exist, and one after
the module closed belongs to the file root.

The lexing is :mod:`rust_source_reading`'s, this suite's one Rust reader. It
already knows where code stops and a comment, string, raw string, byte string
or character literal begins, and its block reader counts nesting. Asking it a
second question is cheaper than growing a second scanner that would eventually
disagree with it: a brace counted inside a literal moves every later
declaration to the wrong depth, and that is exactly the reading this module
must get right.

Re-use policy: import it from `tests/workflow_contracts` only, and read a
declaration from it rather than re-deriving one from `make --dry-run` text.
"""

from __future__ import annotations

import re
import typing as typ

from rust_source_reading import _FUNCTION, _blank, _regions

#: One brace's effect on nesting depth. Everything outside this mapping leaves
#: the depth alone, so the scan counts structure rather than characters.
_BRACE_DELTA: typ.Final[dict[str, int]] = {"{": 1, "}": -1}

#: Any module header, whichever module is wanted. Compiled once rather than per
#: name, because the caller asks about one module in a file that may hold
#: several. A module may be public and may be indented inside another one; the
#: second shape is found here and then rejected by its depth.
_MODULE_HEADER = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?mod[ \t]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*\{",
    re.MULTILINE,
)


def _brace_depths(code: str) -> list[int]:
    """Return the nesting depth at every offset, counted from the file root.

    One pass serves every question below, so a depth is a list lookup rather
    than a rescan per declaration. ``code`` must already have its comments and
    literals blanked: a brace inside either opens or closes nothing, and
    counting one moves every later declaration to the wrong depth.

    Examples
    --------
    >>> _brace_depths("ab{cd}")
    [0, 0, 0, 1, 1, 1]
    """
    depths: list[int] = []
    depth = 0
    for character in code:
        depths.append(depth)
        depth += _BRACE_DELTA.get(character, 0)
    return depths


def _matching_brace(code: str, depths: list[int], open_offset: int) -> int | None:
    """Return the offset of the ``}`` closing the ``{`` at ``open_offset``.

    Depth counting is what stops a nested module or an ``impl`` from ending the
    span early, which would hide every declaration after it. An unbalanced
    source returns None, which the caller reads as "no such module" rather than
    guessing at an extent.

    Examples
    --------
    >>> _matching_brace("ab{cd}", _brace_depths("ab{cd}"), 2)
    5
    >>> _matching_brace("ab{cd", _brace_depths("ab{cd"), 2) is None
    True
    """
    inside_depth = depths[open_offset] + 1
    return next(
        (
            offset
            for offset in range(open_offset + 1, len(depths))
            if code[offset] == "}" and depths[offset] == inside_depth
        ),
        None,
    )


def _module_body(
    code: str, depths: list[int], name: str
) -> tuple[int, int] | None:
    """Return the brace offsets of the top-level ``mod name`` body, or None.

    A module of that name nested inside another is skipped rather than
    returned: its cases have a longer path than the one nextest is given, so
    accepting it would report a selection that cannot happen.
    """
    for header in _MODULE_HEADER.finditer(code):
        if header["name"] != name or depths[header.start()] != 0:
            continue
        open_offset = header.end() - 1
        close_offset = _matching_brace(code, depths, open_offset)
        if close_offset is not None:
            return open_offset, close_offset
    return None


def declared_in_module(source: str, module: str, case: str) -> bool:
    """Is ``case`` declared directly inside a top-level ``mod module``?

    Parameters
    ----------
    source
        The contents of one Rust source file.
    module
        The module segment the path names, normally ``"tests"``.
    case
        The function name the path ends with.

    Examples
    --------
    >>> declared_in_module("mod tests {\\n    fn a() {}\\n}\\n", "tests", "a")
    True
    >>> declared_in_module(
    ...     "mod tests {\\n    mod nested {\\n        fn a() {}\\n    }\\n}\\n",
    ...     "tests",
    ...     "a",
    ... )
    False
    >>> declared_in_module("mod tests {\\n    // fn a() {}\\n}\\n", "tests", "a")
    False

    Returns
    -------
    bool
        True only when the declaration is at the module's own level.
    """
    code = _blank(source, _regions(source))
    depths = _brace_depths(code)
    span = _module_body(code, depths, module)
    if span is None:
        return False
    open_offset, close_offset = span
    # Two conditions, and neither implies the other. The depth test rules out a
    # declaration nested deeper, in a module or an `impl`; the span rules out
    # one at module depth in a *sibling* top-level module, which the depth test
    # would otherwise accept. Both are mutation-probed in the test module.
    return any(
        open_offset < match.start() < close_offset
        and depths[match.start()] == depths[open_offset] + 1
        for match in _FUNCTION.finditer(code)
        if match["name"] == case
    )
