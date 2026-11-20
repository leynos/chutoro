"""Join a CI step's physical lines into the command lines the shell reads.

A workflow step's script is one shell script, and the shell does not read it a
line at a time. A line ending in a backslash continues the current line; a
line ending in a list operator — ``&&``, ``||``, ``|`` or ``|&`` — continues
the current command list, and the next line becomes that operator's
right-hand side. Reading either as a line boundary splits one command into
two, which can hide a skipped installer (``true ||`` on one line, the
installer on the next) or invent a status-bearing last command that the shell
never runs separately.

This module performs both joins, and sets a line's quoted text aside so that
an operator inside quotes is never read as one. ``scheduling_script_reading``
reads the resulting lines as simple commands, so the two concerns stay apart
and neither file outgrows the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_script_reading`` only; this module
reuses the comment and quote scanning in ``scheduling_command_reading``. A new
line-level rule belongs here with the shell behaviour it models stated beside
it.
"""

from __future__ import annotations

import typing as typ

from scheduling_command_reading import closing_quote, strip_shell_comment

#: Operators that leave the command list open at the end of a line. Any of
#: them makes the next line the right-hand side of the list, so the two
#: physical lines are one command for the shell. A lone ``&`` backgrounds the
#: command and opens nothing, and ``;`` ends the list.
_LIST_OPERATORS = ("&&", "||", "|&", "|")

#: The form a quoted word is set aside in while the line's shell syntax is
#: read. Quoted text is data: ``echo '&&' installer`` prints both words, so
#: the quoted operator must not be read as one. The marker holds no operator
#: or comment character, so masking the text keeps it whole through both
#: readings.
_QUOTED = "\x01{}\x02"


class MaskedLine(typ.NamedTuple):
    """A line with its quoted text set aside, and the text that was quoted."""

    text: str
    quoted: list[str]


def mask_quotes(line: str) -> MaskedLine:
    """Return `line` with each quoted word replaced by a marker.

    Quotes decide what is an operator and what is text. ``echo '&&' installer``
    prints both words; once ``shlex`` has removed the quotes, the ``&&`` is
    indistinguishable from an operator, and a reader that splits on operators
    sees a command where the shell ran a word. Setting the quoted text aside
    before the line is read keeps the two apart, and restoring it afterwards
    keeps the words the command is given.

    A backslash escapes the character after it, so an escaped quote opens no
    quoted text. An unterminated quote raises, as in ``strip_shell_comment``.

    Examples
    --------
    >>> mask_quotes("echo 'a b'").quoted
    ['a b']
    >>> mask_quotes("echo 'a b'").text
    'echo \\x010\\x02'
    """
    masked: list[str] = []
    quoted: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            masked.append(line[index : index + 2])
            index += 2
            continue
        if char not in "'\"":
            masked.append(char)
            index += 1
            continue
        end = closing_quote(line, index)
        if end is None:
            message = f"unterminated quote in {line!r}"
            raise ValueError(message)
        quoted.append(line[index + 1 : end])
        masked.append(_QUOTED.format(len(quoted) - 1))
        index = end + 1
    return MaskedLine("".join(masked), quoted)


def restore_quotes(token: str, quoted: list[str]) -> str:
    """Return `token` with each quoted marker replaced by its text.

    Examples
    --------
    >>> restore_quotes("a\\x010\\x02", ["b c"])
    'ab c'
    """
    for position, text in enumerate(quoted):
        token = token.replace(_QUOTED.format(position), text)
    return token


def _escaped(text: str, index: int) -> bool:
    """Return whether the character at `index` is backslash-escaped.

    Examples
    --------
    >>> _escaped("a&b", 1)
    False
    >>> _escaped("a" + chr(92) + "&b", 2)
    True
    >>> _escaped("a" + chr(92) * 2 + "&b", 3)
    False
    """
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _ends_with_operator(line: str) -> bool:
    """Return whether `line` ends with an unescaped list operator.

    Quoted text is data, so an operator inside quotes ends nothing, and a
    backslash makes the character after it literal, so an escaped ``&`` is
    part of a word rather than the start of one.

    Examples
    --------
    >>> _ends_with_operator("make test &&")
    True
    >>> _ends_with_operator("make test |")
    True
    >>> _ends_with_operator("echo '&&'")
    False
    >>> _ends_with_operator("make test &")
    False
    """
    text = mask_quotes(line).text.rstrip()
    for operator in _LIST_OPERATORS:
        if not text.endswith(operator):
            continue
        return not _escaped(text, len(text) - len(operator))
    return False


def _joined_lines(script: str) -> list[str]:
    """Return `script`'s physical lines with backslash continuations joined.

    A workflow script wraps long commands with a trailing backslash. Passing
    one of those lines to ``shlex`` on its own raises ``No escaped
    character``, because the escape has nothing left to escape, so the
    continuations are resolved first.

    Examples
    --------
    The continuation is built with ``chr(92)`` and ``chr(10)``, because a
    docstring holding those escapes would end the example's line:

    >>> _joined_lines(
    ...     "cargo build " + chr(92) + chr(10) + "  --release"
    ... )
    ['cargo build    --release']
    """
    lines: list[str] = []
    pending = ""
    for raw in script.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending = f"{pending}{line[:-1]} "
            continue
        lines.append(f"{pending}{line}")
        pending = ""
    if pending.strip():
        lines.append(pending)
    return lines


def logical_lines(script: str) -> list[str]:
    """Return `script` as the command lines a shell would read.

    Backslash continuations are resolved first. Each resulting line then
    absorbs the next while it ends in a list operator: the shell parses the
    next line as that operator's right-hand side, so ``true ||`` followed by
    an installer line skips the installer, and an installer followed by
    ``&&`` carries its failure into the next line rather than ending as a
    command of its own. Comments are removed before the decision, so a
    trailing comment cannot absorb the next line, and a comment or blank line
    between a continuation and the command it continues is skipped as the
    shell skips it. A comment-only line with nothing to continue stays as an
    empty line, which runs no command.

    Examples
    --------
    >>> logical_lines("installer &&" + chr(10) + "true")
    ['installer && true']
    >>> logical_lines("true ||" + chr(10) + "installer")
    ['true || installer']
    >>> logical_lines("installer " + chr(35) + " note")
    ['installer']
    >>> logical_lines("installer " + chr(35) + " note" + chr(10) + "true")
    ['installer', 'true']
    >>> logical_lines(
    ...     "installer && " + chr(35) + " note" + chr(10) + "true"
    ... )
    ['installer && true']
    """
    lines: list[str] = []
    pending = ""
    for line in _joined_lines(script):
        stripped = strip_shell_comment(line).strip()
        if pending and not stripped:
            continue
        pending = f"{pending} {stripped}" if pending else stripped
        if _ends_with_operator(pending):
            continue
        lines.append(pending)
        pending = ""
    if pending.strip():
        lines.append(pending)
    return lines