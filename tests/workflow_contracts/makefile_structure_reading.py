"""Read a Makefile's directive structure around its declarations.

GNU make does not read a Makefile line by line. A comment runs past a trailing
backslash, a ``define`` body is variable text rather than directives, and a
conditional's body declares only when its condition holds. A ``.PHONY`` line
inside any of those is not evidence that the target cannot be skipped by a
same-named file, so the readers here drop such text before the declarations
are collected. Continuations are joined afterwards, as make joins them.

Separating the structure from the phony reading keeps both files inside the
400-line limit ``AGENTS.md`` sets, and states the make behaviour beside the
reader that models it rather than beside the target it protects.

Re-use policy: import from ``makefile_reading`` only, which applies these
readings to the ``.PHONY`` declarations it collects. A new structural rule
belongs here with the make behaviour it models stated beside it.
"""

from __future__ import annotations


def make_comment_start(line: str) -> int | None:
    """Return the index of the first unescaped ``#`` in `line`, if any.

    A Make comment starts at an unescaped ``#`` anywhere on the line, not only
    at a word start as a shell comment does, and ``\\#`` is a literal ``#`` in
    a target name.

    Examples
    --------
    >>> make_comment_start("clean # tidy up")
    6
    >>> escaped = "clean " + chr(92) + "#1"
    >>> make_comment_start(escaped) is None
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


def _directive(line: str) -> str:
    """Return `line` without its Make comment, for reading a directive.

    Examples
    --------
    >>> _directive("define job " + chr(35) + " note")
    'define job '
    >>> _directive("define job")
    'define job'
    """
    comment = make_comment_start(line)
    return line if comment is None else line[:comment]


def _opens_a_definition(line: str) -> bool:
    """Return whether `line` opens a ``define`` body.

    A ``define`` names a variable whose value is every following line up to
    ``endef``, so a ``.PHONY`` written inside one is variable text rather than
    a declaration. ``override define`` and ``export define`` open one too. A
    recipe line begins with a tab and is not a directive, and a commented-out
    ``define`` opens nothing.

    Examples
    --------
    >>> _opens_a_definition("define UNUSED")
    True
    >>> _opens_a_definition("override define job")
    True
    >>> _opens_a_definition(chr(9) + "define job")
    False
    >>> _opens_a_definition("# define job")
    False
    """
    if line[:1] == "\t":
        return False
    words = _directive(line).split()
    return words[:1] == ["define"] or words[:2] in (
        ["override", "define"],
        ["export", "define"],
    )


def _definition_delta(line: str) -> int:
    """Return how `line` changes the ``define`` nesting depth.

    GNU make lets a definition hold another one, so a body ends at the
    ``endef`` that matches its own ``define`` rather than at the first one.

    Examples
    --------
    >>> _definition_delta("define INNER")
    1
    >>> _definition_delta("endef")
    -1
    >>> _definition_delta(".PHONY: gate")
    0
    """
    if _opens_a_definition(line):
        return 1
    return -1 if _closes_a_definition(line) else 0


def _opens_a_conditional(line: str) -> bool:
    """Return whether `line` opens a conditional block.

    A ``.PHONY`` inside an ``if``/``else`` block declares only when that
    branch is taken, and the reading does not evaluate conditions, so the
    declaration is not evidence of an unconditional target.

    Examples
    --------
    >>> _opens_a_conditional("ifeq (0,1)")
    True
    >>> _opens_a_conditional("ifdef NAME")
    True
    >>> _opens_a_conditional("else")
    False
    >>> _opens_a_conditional(chr(9) + "ifeq (0,1)")
    False
    """
    if line[:1] == "\t":
        return False
    word = _directive(line).split()[:1]
    return word in (["ifeq"], ["ifneq"], ["ifdef"], ["ifndef"])


def _conditional_delta(line: str) -> int:
    """Return how `line` changes the conditional nesting depth.

    ``else`` and ``endif`` close no conditional of their own: ``else`` stays
    inside the one it branches, and ``endif`` closes one level. A tab-indented
    line is recipe text, so it closes nothing.

    Examples
    --------
    >>> _conditional_delta("ifeq (0,1)")
    1
    >>> _conditional_delta("else")
    0
    >>> _conditional_delta("endif")
    -1
    >>> _conditional_delta(chr(9) + "endif")
    0
    >>> _conditional_delta(".PHONY: gate")
    0
    """
    if line[:1] == "\t":
        return 0
    if _opens_a_conditional(line):
        return 1
    if _directive(line).split()[:1] == ["endif"]:
        return -1
    return 0


def _closes_a_definition(line: str) -> bool:
    """Return whether `line` ends a ``define`` body.

    Examples
    --------
    >>> _closes_a_definition("endef")
    True
    >>> _closes_a_definition("endef " + chr(35) + " note")
    True
    >>> _closes_a_definition("echo endef")
    False
    >>> _closes_a_definition(chr(9) + "endef")
    False
    """
    if line[:1] == "\t":
        return False
    return _directive(line).split()[:1] == ["endef"]


def _logical_lines(text: str) -> list[str]:
    """Return `text`'s physical lines with Make continuations joined.

    GNU make replaces a backslash-newline pair with a space before it reads a
    directive, so a line that merely continues an assignment cannot open or
    close a conditional or a definition. Only an unescaped trailing backslash
    joins.

    Examples
    --------
    >>> joined = _logical_lines(".PHONY: clean " + chr(92) + chr(10) + " gate")
    >>> " ".join(joined[0].split())
    '.PHONY: clean gate'
    >>> doubled = "a" + chr(92) * 2 + chr(10) + "b"
    >>> len(_logical_lines(doubled))
    2
    >>> _logical_lines("x " + chr(92) + chr(10) + "endef")[0].endswith("endef")
    True
    """
    lines: list[str] = []
    pending = ""
    for physical in text.splitlines():
        joined = f"{pending} {physical}" if pending else physical
        if _continues(physical):
            pending = joined[:-1]
            continue
        lines.append(joined)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def _conditional_after(
    line: str, conditionals: int, definition: int
) -> tuple[int, int]:
    """Return the conditional and inner-definition depths after `line`.

    A ``define`` opened inside an inactive conditional hides everything up to
    its own ``endef``, so the ``endif`` written there is variable text rather
    than the end of the branch.

    Examples
    --------
    >>> _conditional_after(".PHONY: gate", 1, 0)
    (1, 0)
    >>> _conditional_after("define UNUSED", 1, 0)
    (1, 1)
    >>> _conditional_after("endif", 1, 1)
    (1, 1)
    >>> _conditional_after("endif", 1, 0)
    (0, 0)
    """
    if definition:
        return conditionals, definition + _definition_delta(line)
    if _opens_a_definition(line):
        return conditionals, 1
    return conditionals + _conditional_delta(line), 0


def declaring_lines(text: str) -> list[str]:
    """Return `text`'s lines outside a recipe, definition or conditional.

    Make joins a continued line before it reads a directive, so the lines are
    read logically. A recipe line begins with a tab, a ``define`` body is
    stored in its variable, and a conditional's body declares only when its
    condition holds; none of them declares a phony target. Comment lines are
    kept for the caller to remove, because a ``#`` after a completed line is
    not a declaration either.

    Examples
    --------
    >>> declaring_lines(".PHONY: gate " + chr(35) + " note")
    ['.PHONY: gate # note']
    >>> continued = "# off " + chr(92) + chr(10) + ".PHONY: gate"
    >>> " ".join(declaring_lines(continued)[0].split())
    '# off .PHONY: gate'
    >>> doubled = "# off " + chr(92) * 2 + chr(10) + ".PHONY: gate"
    >>> declaring_lines(doubled)[1]
    '.PHONY: gate'
    >>> declaring_lines("gate:" + chr(10) + chr(9) + ".PHONY: gate")
    ['gate:']
    >>> hidden = (
    ...     "ifeq (1,1)" + chr(10) + "define UNUSED" + chr(10) + "endif"
    ...     + chr(10) + ".PHONY: gate" + chr(10) + "endef" + chr(10) + "endif"
    ... )
    >>> declaring_lines(hidden)
    []
    >>> continued_comment = (
    ...     "ifeq (0,1)" + chr(10) + "# hidden " + chr(92) + chr(10)
    ...     + "endif" + chr(10) + ".PHONY: gate" + chr(10) + "endif"
    ... )
    >>> declaring_lines(continued_comment)
    []
    >>> continued_assignment = (
    ...     "ifeq (0,1)" + chr(10) + "ignored = " + chr(92) + chr(10)
    ...     + "endif" + chr(10) + ".PHONY: gate" + chr(10) + "endif"
    ...     + chr(10) + "gate: ; @:"
    ... )
    >>> declaring_lines(continued_assignment)
    ['gate: ; @:']
    >>> joined_endef = (
    ...     "define UNUSED" + chr(10) + "text " + chr(92) + chr(10)
    ...     + "endef" + chr(10) + ".PHONY: gate" + chr(10) + "endef"
    ... )
    >>> declaring_lines(joined_endef)
    []
    """
    lines: list[str] = []
    definition = 0
    conditionals = 0
    conditional_definition = 0
    for line in _logical_lines(text):
        if line[:1] == "\t":
            continue
        if definition:
            definition += _definition_delta(line)
            continue
        if conditionals:
            conditionals, conditional_definition = _conditional_after(
                line, conditionals, conditional_definition
            )
            continue
        if _opens_a_definition(line):
            definition = 1
            continue
        if _opens_a_conditional(line):
            conditionals = 1
            continue
        lines.append(line)
    return lines


def declarations(lines: list[str]) -> list[str]:
    """Return each declaration in `lines`, continuations joined.

    Make joins a line ending in an unescaped backslash to the next one before
    it reads either, so the joined line is what a declaration is. The join
    uses a space, as make's does outside a recipe. A doubled backslash ends
    the declaration, so the next line is a declaration of its own.

    Examples
    --------
    >>> joined = declarations([".PHONY: clean " + chr(92), " gate"])
    >>> " ".join(joined[0].split())
    '.PHONY: clean gate'
    >>> doubled = declarations([".PHONY: clean " + chr(92) * 2, ".PHONY: g"])
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
