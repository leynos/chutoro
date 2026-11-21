"""Set a shell line's quoted and escaped text aside for safe reading.

A shell line has to be read twice: once as syntax, to find the operators and
comments, and once as words, to recover what the command is given. Quotes
serve the first reading by hiding the second: ``echo '&&' installer`` prints
both words, but once the quotes are removed the ``&&`` is indistinguishable
from an operator. A backslash does the same to a bare character. Marking both
kinds of text before the line is read, and restoring them afterwards, keeps
the two readings apart.

This module holds the masking alone, so ``scheduling_command_reading`` can
read comments and operators over the masked text. The marker and the
restoration are both here because they are two halves of one encoding.

Re-use policy: import from ``scheduling_command_reading``,
``scheduling_here_document_reading``, ``scheduling_line_reading`` and
``scheduling_segment_reading``, which read the masked text. A new piece of
masked syntax belongs here with the reason it must be hidden stated beside it.
"""

from __future__ import annotations

import typing as typ

from scheduling_command_reading import closing_quote

#: The form a quoted word is set aside in while a line's shell syntax is read.
#: The marker holds no operator or comment character, so masking the text keeps
#: it whole through every later reading.
_QUOTED = "\x01{}\x02"


class MaskedLine(typ.NamedTuple):
    """A line with its quoted and escaped text set aside for safe reading."""

    text: str
    quoted: list[str]


def _take_quoted(line: str, index: int) -> tuple[str, int]:
    """Return a quoted word's text and the index just past its closing quote.

    The text is the word's content as written; an unterminated quote is not a
    command the shell runs, so it raises rather than reading past the end.

    Examples
    --------
    >>> _take_quoted("'a b' c", 0)
    ('a b', 5)
    >>> _take_quoted('"a b" c', 0)
    ('a b', 5)
    >>> _take_quoted("'unclosed", 0)
    Traceback (most recent call last):
        ...
    ValueError: unterminated quote in "'unclosed"
    """
    end = closing_quote(line, index)
    if end is None:
        message = f"unterminated quote in {line!r}"
        raise ValueError(message)
    return line[index + 1 : end], end + 1


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
    >>> ansi = mask_quotes("set $'-n'")
    >>> ansi.quoted
    ['-n']
    >>> "$" in ansi.text
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
        if char == "$" and line[index + 1 : index + 2] == "'":
            # An ANSI-C string is one word, dollar sign and quotes included,
            # so the dollar is left out of the masked text.
            index += 1
        elif char not in "'\"":
            masked.append(char)
            index += 1
            continue
        text, index = _take_quoted(line, index)
        quoted.append(text)
        masked.append(_QUOTED.format(len(quoted) - 1))
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
