"""Read a Make declaration for the targets it names.

The scheduling contract checks that its target is declared ``.PHONY``, because
a same-named file would otherwise let ``make`` report the target as up to date
and skip the gate. Reading that declaration is Make syntax rather than shell
syntax: a comment starts at an unescaped ``#`` anywhere on the line, a
backslash continues a declaration onto the next line, and a comment ending in
a backslash continues too, so the line after it declares nothing.

This module holds that reading. ``scheduling_reading`` reads the recipe the
target runs, and the two are separate because reading both pushed
``scheduling_reading`` past the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_contract_test`` only, the one place
that asserts over it. A new Make rule belongs here with the make behaviour it
models stated beside it.
"""

from __future__ import annotations

import re


def _make_comment_start(line: str) -> int | None:
    """Return the index of the first unescaped ``#`` in `line`, if any.

    A Make comment starts at an unescaped ``#`` anywhere on the line, not only
    at a word start as a shell comment does, and ``\\#`` is a literal ``#`` in
    a target name.

    Examples
    --------
    >>> _make_comment_start("clean # tidy up")
    6
    >>> escaped = "clean " + chr(92) + "#1"
    >>> _make_comment_start(escaped) is None
    True
    """
    index = 0
    while index < len(line):
        if line[index] == "\\" and index + 1 < len(line):
            index += 2
            continue
        if line[index] == "#":
            return index
        index += 1
    return None


def _comment_continues(line: str) -> bool:
    """Return whether a Make comment runs past the end of `line`.

    GNU make continues a comment whose line ends in a backslash, so the whole
    next line is comment text and declares no target.

    Examples
    --------
    >>> _comment_continues("clean # note " + chr(92))
    True
    >>> _comment_continues("clean # note")
    False
    """
    return line.rstrip().endswith("\\")


def _declared_names(line: str) -> list[str]:
    """Return the target names one declaration line adds.

    The continuation backslash ends the line rather than naming a target, an
    escaped ``#`` names a target that contains one, and anything from an
    unescaped ``#`` on is a Make comment rather than a name.

    Examples
    --------
    >>> _declared_names(" clean " + chr(92))
    ['clean']
    >>> _declared_names(" clean" + chr(92) + "#1")
    ['clean#1']
    >>> _declared_names(" clean # tidy up")
    ['clean']
    """
    comment = _make_comment_start(line)
    uncommented = line if comment is None else line[:comment]
    return uncommented.rstrip().removesuffix("\\").replace("\\#", "#").split()


def phony_targets(text: str) -> set[str]:
    """Return the names a ``.PHONY`` declaration covers.

    GNU make collapses continued declarations, so each declaration is read
    line by line and a continuation backslash only joins the next line's
    names. Comments are removed first: ``.PHONY: clean # gate`` declares
    ``clean`` alone, and reading the comment would make a same-named file seem
    covered when the target could still be skipped. A comment ending in a
    backslash continues onto the next line, which declares nothing either, so
    ``.PHONY: clean # \\`` followed by ``gate`` still declares only ``clean``.

    Examples
    --------
    >>> sorted(phony_targets(".PHONY: clean " + chr(35) + " gate"))
    ['clean']
    >>> declaration = ".PHONY: clean " + chr(92) + chr(10) + " gate"
    >>> sorted(phony_targets(declaration))
    ['clean', 'gate']
    >>> continued = (
    ...     ".PHONY: clean " + chr(35) + " " + chr(92) + chr(10) + " gate"
    ... )
    >>> sorted(phony_targets(continued))
    ['clean']
    """
    declared = re.findall(r"^\.PHONY:((?:.*\\\n)*.*)$", text, re.MULTILINE)
    names: set[str] = set()
    for declaration in declared:
        in_comment = False
        for line in declaration.splitlines():
            if in_comment:
                in_comment = _comment_continues(line)
                continue
            if _make_comment_start(line) is not None:
                in_comment = _comment_continues(line)
            names.update(_declared_names(line))
    return names
