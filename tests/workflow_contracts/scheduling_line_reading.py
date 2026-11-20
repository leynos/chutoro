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
an operator inside quotes is never read as one. It also drops here-document
bodies, whose text is standard input for the command that opened them rather
than a command line of its own. ``scheduling_script_reading`` reads the
resulting lines as simple commands, so the two concerns stay apart and neither
file outgrows the 400-line limit ``AGENTS.md`` sets.

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

#: Characters that end a here-document's delimiter word. The shell lets a
#: metacharacter follow it, as ``cat <<EOF; make test`` does, so the word ends
#: where the operator begins.
_DELIMITER_ENDS = frozenset(" \t;&|()<>")


class MaskedLine(typ.NamedTuple):
    """A line with its quoted and escaped text set aside for safe reading."""

    text: str
    quoted: list[str]


def mask_quotes(line: str) -> MaskedLine:
    """Return `line` with each quoted word and escaped character set aside.

    Quotes decide what is an operator and what is text. ``echo '&&' installer``
    prints both words; once ``shlex`` has removed the quotes, the ``&&`` is
    indistinguishable from an operator, and a reader that splits on operators
    sees a command where the shell ran a word. A backslash does the same to a
    bare character: ``echo \\&\\& installer`` prints the two ampersands, and
    ``shlex`` removes the escapes before a separator check can tell the
    difference. Setting both kinds of text aside before the line is read keeps
    words and operators apart, and restoring them afterwards keeps the words
    the command is given.

    A backslash escapes the character after it, so an escaped quote opens no
    quoted text. An unterminated quote raises, as in ``strip_shell_comment``.

    Examples
    --------
    >>> mask_quotes("echo 'a b'").quoted
    ['a b']
    >>> mask_quotes("echo 'a b'").text
    'echo \\x010\\x02'
    >>> escaped = mask_quotes("echo " + chr(92) + "&" + chr(92) + "& x")
    >>> escaped.quoted
    ['&', '&']
    >>> "&" in escaped.text
    False
    """
    masked: list[str] = []
    quoted: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            quoted.append(line[index + 1])
            masked.append("\\" + _QUOTED.format(len(quoted) - 1))
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


def _ends_with_operator(line: str) -> bool:
    """Return whether `line` ends with an unescaped list operator.

    Quoted text is data, so an operator inside quotes ends nothing, and a
    backslash makes the character after it literal, so an escaped ``&`` is
    part of a word rather than the start of one. Masking both before the
    check leaves only operators the shell would act on.

    Examples
    --------
    >>> _ends_with_operator("make test &&")
    True
    >>> _ends_with_operator("make test |")
    True
    >>> _ends_with_operator("echo '&&'")
    False
    >>> _ends_with_operator("make test " + chr(92) + "&&")
    False
    >>> _ends_with_operator("make test &")
    False
    """
    text = mask_quotes(line).text.rstrip()
    return any(text.endswith(operator) for operator in _LIST_OPERATORS)


def _joined_lines(script: str) -> list[str]:
    """Return `script`'s physical lines with backslash continuations joined.

    A workflow script wraps long commands with a trailing backslash. Passing
    one of those lines to ``shlex`` on its own raises ``No escaped
    character``, because the escape has nothing left to escape, so the
    continuations are resolved first.

    The shell deletes an unquoted backslash-newline pair and inserts nothing
    in its place, so the two parts of a split word are joined exactly. A space
    inserted here would turn the installer path ``scripts/install-\\`` +
    ``nextest.sh`` into two words the step never runs.

    Examples
    --------
    The continuation is built with ``chr(92)`` and ``chr(10)``, because a
    docstring holding those escapes would end the example's line:

    >>> _joined_lines(
    ...     "cargo build " + chr(92) + chr(10) + "  --release"
    ... )
    ['cargo build   --release']
    >>> _joined_lines(
    ...     "scripts/install-" + chr(92) + chr(10) + "nextest.sh"
    ... )
    ['scripts/install-nextest.sh']
    """
    lines: list[str] = []
    pending = ""
    for raw in script.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending = f"{pending}{line[:-1]}"
            continue
        lines.append(f"{pending}{line}")
        pending = ""
    if pending.strip():
        lines.append(pending)
    return lines


def _delimiter_start(masked: str, start: int) -> int:
    """Return the index where a ``<<`` operator's delimiter word begins.

    ``<<-`` strips leading tabs from the body, and either form may put spaces
    between the operator and the word.

    Examples
    --------
    >>> _delimiter_start("<<EOF", 2)
    2
    >>> _delimiter_start("<<-EOF", 2)
    3
    >>> _delimiter_start("<< EOF", 2)
    3
    """
    if masked.startswith("-", start):
        start += 1
    while start < len(masked) and masked[start] in " \t":
        start += 1
    return start


def _delimiter_end(masked: str, start: int) -> int:
    """Return the index where the delimiter word beginning at `start` ends.

    Examples
    --------
    >>> _delimiter_end("EOF; make test", 0)
    3
    >>> _delimiter_end("EOF", 0)
    3
    """
    end = start
    while end < len(masked) and masked[end] not in _DELIMITER_ENDS:
        end += 1
    return end


def here_document_terminators(line: str) -> list[str]:
    """Return the words that end the here-documents `line` opens.

    A here-document's body is standard input for the command that reads it,
    not a command the shell runs. Read as commands, a body that named the
    installer looked like an installation, so ``cat <<'EOF'`` alone satisfied
    an ordering contract that ran nothing. Quoted text is data, so a ``<<``
    inside quotes opens nothing; the delimiter itself may be quoted, and the
    quoted form is restored because that is the word the closing line repeats.

    A ``<<`` in an arithmetic expansion is read as an opening too, which can
    only make later commands look like body text: the reading then finds no
    evidence rather than false evidence, so the mistake fails the contract
    instead of passing it.

    Examples
    --------
    >>> here_document_terminators("cat <<'EOF'")
    ['EOF']
    >>> here_document_terminators("cat <<-EOF")
    ['EOF']
    >>> here_document_terminators("echo '<<EOF'")
    []
    >>> here_document_terminators("cat <<EOF; make test")
    ['EOF']
    >>> here_document_terminators("cat <<A <<B")
    ['A', 'B']
    """
    masked = mask_quotes(line)
    terminators: list[str] = []
    index = masked.text.find("<<")
    while index != -1:
        start = _delimiter_start(masked.text, index + 2)
        end = _delimiter_end(masked.text, start)
        if end > start:
            terminators.append(
                restore_quotes(masked.text[start:end], masked.quoted)
            )
        index = masked.text.find("<<", max(end, index + 2))
    return terminators


def _consumes_body_line(line: str, terminators: list[str]) -> bool:
    """Return whether `line` belongs to a here-document body, consuming it.

    `terminators` holds the words that end the documents still open, outermost
    first. The line that ends the current body closes the document and is
    consumed with it, so the next line is read as a command again.

    Examples
    --------
    >>> open_documents = ["EOF"]
    >>> _consumes_body_line("installer", open_documents)
    True
    >>> open_documents
    ['EOF']
    >>> _consumes_body_line("EOF", open_documents)
    True
    >>> open_documents
    []
    >>> _consumes_body_line("installer", open_documents)
    False
    """
    if not terminators:
        return False
    if line.strip() == terminators[0]:
        terminators.pop(0)
    return True


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
    empty line, which runs no command. A here-document's body is not a command
    line either: the line that opened it names the word that ends the body,
    and every line up to and including that word is dropped.

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
    >>> logical_lines("cat <<'EOF'" + chr(10) + "installer" + chr(10) + "EOF")
    ["cat <<'EOF'"]
    >>> logical_lines(
    ...     "cat <<'EOF'" + chr(10) + "installer" + chr(10) + "EOF"
    ...     + chr(10) + "installer"
    ... )
    ["cat <<'EOF'", 'installer']
    """
    lines: list[str] = []
    pending = ""
    terminators: list[str] = []
    for line in _joined_lines(script):
        if _consumes_body_line(line, terminators):
            continue
        stripped = strip_shell_comment(line).strip()
        if pending and not stripped:
            continue
        pending = f"{pending} {stripped}" if pending else stripped
        if _ends_with_operator(pending):
            continue
        lines.append(pending)
        terminators = here_document_terminators(pending)
        pending = ""
    if pending.strip():
        lines.append(pending)
    return lines
