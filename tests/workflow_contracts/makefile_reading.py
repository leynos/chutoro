"""Read a Make declaration for the targets it names.

The scheduling contract checks that its target is declared ``.PHONY``, because
a same-named file would otherwise let ``make`` report the target as up to date
and skip the gate. Reading that declaration is Make syntax rather than shell
syntax: a comment starts at an unescaped ``#`` anywhere on the line, a
backslash continues a declaration onto the next line, and a comment ending in
an unescaped backslash continues too, so the line after it declares nothing.
A doubled backslash leaves a literal one and continues nothing.

This module holds that reading. ``scheduling_reading`` reads the recipe the
target runs, and the two are separate because reading both pushed
``scheduling_reading`` past the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_contract_test`` only, the one place
that asserts over it. A new Make rule belongs here with the make behaviour it
models stated beside it.
"""

from __future__ import annotations

#: The word that opens a phony-target declaration.
_PHONY = ".PHONY:"


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


def _continues(line: str) -> bool:
    """Return whether `line` ends in a backslash that continues it.

    GNU make joins a line ending in a backslash to the next one, whether the
    line declares a target or holds a comment. Only an unescaped backslash
    joins: a doubled one leaves a literal backslash and ends the line, and a
    backslash followed by a space escapes the space rather than the newline,
    so nothing is stripped before the count.

    Examples
    --------
    >>> _continues("clean " + chr(92))
    True
    >>> _continues("clean")
    False
    >>> _continues("clean " + chr(92) * 2)
    False
    >>> _continues("clean " + chr(92) * 3)
    True
    >>> _continues("clean " + chr(92) + chr(32))
    False
    """
    run = len(line) - len(line.rstrip("\\"))
    return run % 2 == 1


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


def _declaring_lines(text: str) -> list[str]:
    """Return `text`'s lines outside a continued Make comment.

    GNU make continues a comment whose line ends in an unescaped backslash, so
    the line after it is comment text and declares nothing however it reads.
    Dropping those lines before the declarations are selected keeps a
    commented-out ``.PHONY:`` line from declaring a target that nothing
    covers. The backslash that continued the comment is removed with them: it
    continued the comment rather than the declaration, and leaving it would
    make the selection glue the next real declaration onto this one. A doubled
    backslash ends the comment instead, so the line after it stays.

    Examples
    --------
    >>> lines = _declaring_lines("# off " + chr(92) + chr(10) + ".PHONY: gate")
    >>> lines == ["# off "]
    True
    >>> doubled = "# off " + chr(92) * 2 + chr(10) + ".PHONY: gate"
    >>> len(_declaring_lines(doubled))
    2
    >>> _declaring_lines(doubled)[1]
    '.PHONY: gate'
    >>> _declaring_lines(".PHONY: gate " + chr(35) + " note")
    ['.PHONY: gate # note']
    """
    lines: list[str] = []
    in_comment = False
    for line in text.splitlines():
        if in_comment:
            in_comment = _continues(line)
            continue
        continues = _make_comment_start(line) is not None and _continues(line)
        in_comment = continues
        lines.append(line[:-1] if continues else line)
    return lines


def _declarations(lines: list[str]) -> list[str]:
    """Return each declaration in `lines`, continuations joined.

    Make joins a line ending in an unescaped backslash to the next one before
    it reads either, so the joined line is what a declaration is. The join
    uses a space, as make's does outside a recipe. A doubled backslash ends
    the declaration, so the next line is a declaration of its own.

    Examples
    --------
    >>> joined = _declarations([".PHONY: clean " + chr(92), " gate"])
    >>> " ".join(joined[0].split())
    '.PHONY: clean gate'
    >>> doubled = _declarations([".PHONY: clean " + chr(92) * 2, ".PHONY: g"])
    >>> len(doubled)
    2
    >>> doubled[1]
    '.PHONY: g'
    """
    declarations: list[str] = []
    pending = ""
    for line in lines:
        joined = f"{pending} {line}" if pending else line
        if _continues(line):
            pending = joined[:-1]
            continue
        declarations.append(joined)
        pending = ""
    if pending:
        declarations.append(pending)
    return declarations


def phony_targets(text: str) -> set[str]:
    """Return the names a ``.PHONY`` declaration covers.

    Make joins a continued declaration into one line before it reads the
    comment on it, so the declaration is collapsed first and the comment is
    then removed: ``.PHONY: clean # gate`` declares ``clean`` alone, and
    reading the comment would make a same-named file seem covered when the
    target could still be skipped. A comment ending in an unescaped backslash
    continues onto the next line, which declares nothing either, so
    ``.PHONY: clean # \\`` followed by ``gate`` still declares only ``clean``;
    a doubled backslash ends the comment, so the next line declares as usual.

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
    >>> commented = "# off " + chr(92) + chr(10) + ".PHONY: gate"
    >>> phony_targets(commented)
    set()
    >>> resumed = (
    ...     ".PHONY: clean " + chr(35) + " " + chr(92) + chr(10) + "gone"
    ...     + chr(10) + ".PHONY: gate"
    ... )
    >>> sorted(phony_targets(resumed))
    ['clean', 'gate']
    >>> doubled = (
    ...     ".PHONY: clean " + chr(35) + " " + chr(92) * 2 + chr(10)
    ...     + ".PHONY: gate"
    ... )
    >>> sorted(phony_targets(doubled))
    ['clean', 'gate']
    """
    names: set[str] = set()
    for declaration in _declarations(_declaring_lines(text)):
        comment = _make_comment_start(declaration)
        if comment is not None:
            declaration = declaration[:comment]
        if declaration.startswith(_PHONY):
            names.update(_declared_names(declaration[len(_PHONY) :]))
    return names