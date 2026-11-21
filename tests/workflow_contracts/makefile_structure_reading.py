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


def _starts_a_continued_comment(line: str) -> bool:
    """Return whether `line` is a comment that continues onto the next one.

    Examples
    --------
    >>> _starts_a_continued_comment("# note " + chr(92))
    True
    >>> _starts_a_continued_comment(".PHONY: gate")
    False
    """
    return make_comment_start(line) is not None and _continues(line)


def _keep_line(lines: list[str], line: str) -> bool:
    """Append `line`, dropping a continuation backslash, and report a comment.

    A continued comment is kept without its backslash: the backslash
    continued the comment rather than the declaration that follows it.

    Examples
    --------
    >>> kept = []
    >>> _keep_line(kept, ".PHONY: gate")
    False
    >>> kept
    ['.PHONY: gate']
    >>> kept = []
    >>> _keep_line(kept, "# note " + chr(92))
    True
    >>> kept == ["# note "]
    True
    """
    if _starts_a_continued_comment(line):
        lines.append(line[:-1])
        return True
    lines.append(line)
    return False


def _definition_step(
    line: str, depth: int, continued: bool
) -> tuple[int, bool]:
    """Return the definition depth and continuation state after `line`.

    A line that continues joins the next one before make reads either, so an
    ``endef`` written after a continuation is part of the joined text rather
    than a directive of its own.

    Examples
    --------
    >>> _definition_step("text " + chr(92), 1, False)
    (1, True)
    >>> _definition_step("endef", 1, True)
    (1, False)
    >>> _definition_step("endef", 1, False)
    (0, False)
    """
    if continued or _continues(line):
        return depth, _continues(line)
    return depth + _definition_delta(line), False


def declaring_lines(text: str) -> list[str]:
    """Return `text`'s lines outside a continued comment or a definition.

    GNU make continues a comment whose line ends in an unescaped backslash, so
    the line after it is comment text and declares nothing however it reads.
    Dropping those lines before the declarations are selected keeps a
    commented-out ``.PHONY:`` line from declaring a target that nothing
    covers. The backslash that continued the comment is removed with them: it
    continued the comment rather than the declaration, and leaving it would
    make the selection glue the next real declaration onto this one. A doubled
    backslash ends the comment instead, so the line after it stays. A
    ``define`` body is dropped for the same reason: make stores its lines in
    the named variable, so a ``.PHONY`` written there declares nothing. A
    conditional block is dropped as well: the reading does not evaluate its
    condition, so a declaration inside it is not evidence of an unconditional
    target.

    Examples
    --------
    >>> lines = declaring_lines("# off " + chr(92) + chr(10) + ".PHONY: gate")
    >>> lines == ["# off "]
    True
    >>> declaring_lines(".PHONY: gate " + chr(35) + " note")
    ['.PHONY: gate # note']
    >>> nested = "define OUTER" + chr(10) + "define INNER" + chr(10) + "endef"
    >>> declaring_lines(nested + chr(10) + ".PHONY: gate" + chr(10) + "endef")
    []
    >>> continued = (
    ...     "ifeq (0,1)" + chr(10) + "# hidden " + chr(92) + chr(10)
    ...     + "endif" + chr(10) + ".PHONY: gate" + chr(10) + "endif"
    ... )
    >>> declaring_lines(continued)
    []
    >>> joined = (
    ...     "define UNUSED" + chr(10) + "text " + chr(92) + chr(10)
    ...     + "endef" + chr(10) + ".PHONY: gate" + chr(10) + "endef"
    ... )
    >>> declaring_lines(joined)
    []
    """
    lines: list[str] = []
    in_comment = False
    definition = 0
    definition_continued = False
    conditionals = 0
    for line in text.splitlines():
        if in_comment:
            in_comment = _continues(line)
            continue
        if definition:
            definition, definition_continued = _definition_step(
                line, definition, definition_continued
            )
            continue
        if conditionals:
            in_comment = _starts_a_continued_comment(line)
            conditionals += _conditional_delta(line)
            continue
        if _opens_a_definition(line):
            definition = 1
            continue
        if _opens_a_conditional(line):
            conditionals = 1
            continue
        in_comment = _keep_line(lines, line)
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
