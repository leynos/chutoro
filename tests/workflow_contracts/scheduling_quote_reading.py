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


#: The single-character ANSI-C escapes, as the shell resolves them.
_ANSI_C_SIMPLE = {
    "a": chr(7),
    "b": chr(8),
    "e": chr(27),
    "f": chr(12),
    "n": chr(10),
    "r": chr(13),
    "t": chr(9),
    "v": chr(11),
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
}


#: The hexadecimal digits an ANSI-C code escape is written with.
_HEX_DIGITS = "0123456789abcdefABCDEF"


def _digits(text: str, allowed: str) -> str:
    """Return the leading run of `text` made of characters in `allowed`.

    Examples
    --------
    >>> _digits("2dn", "0123456789abcdef")
    '2d'
    >>> _digits("xyz", "0123456789")
    ''
    """
    end = 0
    while end < len(text) and text[end] in allowed:
        end += 1
    return text[:end]


def _numeric_escape(text: str, index: int) -> tuple[str | None, int]:
    """Return the character a numeric ANSI-C escape at `index` denotes.

    Bash reads ``\\xHH`` and ``\\uHHHH``/``\\UHHHHHHHH`` as one character by its
    hexadecimal code, and ``\\NNN`` as up to three octal digits. ``None``
    reports that the text at `index` is not a complete escape; the index is
    then unchanged.

    Examples
    --------
    >>> _numeric_escape(chr(92) + "x41", 0)
    ('A', 4)
    >>> _numeric_escape(chr(92) + "u006e", 0)
    ('n', 6)
    >>> _numeric_escape(chr(92) + "101", 0)
    ('A', 4)
    >>> _numeric_escape(chr(92) + "q", 0)[0] is None
    True
    """
    if text[index : index + 1] != "\\":
        return None, index
    marker = text[index + 1 : index + 2]
    if marker in {"x", "u", "U"}:
        return _hex_escape(text, index)
    if marker and marker in "01234567":
        digits = _digits(text[index + 1 : index + 4], "01234567")
        return chr(int(digits, 8) & 0xFF), index + 1 + len(digits)
    return None, index


def _hex_escape(text: str, index: int) -> tuple[str | None, int]:
    """Return the character a ``\\x``, ``\\u`` or ``\\U`` escape denotes.

    ``\\x`` takes up to two hexadecimal digits; ``\\u`` takes four and ``\\U``
    takes eight, and a short one is not an escape the shell reads.

    Examples
    --------
    >>> _hex_escape(chr(92) + "xe9", 0)
    ('é', 4)
    >>> _hex_escape(chr(92) + "u006e", 0)
    ('n', 6)
    >>> _hex_escape(chr(92) + "u06e", 0)[0] is None
    True
    """
    marker = text[index + 1 : index + 2]
    if marker == "x":
        digits = _digits(text[index + 2 : index + 4], _HEX_DIGITS)
        if not digits:
            return None, index
        return chr(int(digits, 16)), index + 2 + len(digits)
    width = 4 if marker == "u" else 8
    digits = _digits(text[index + 2 : index + 2 + width], _HEX_DIGITS)
    if len(digits) != width:
        return None, index
    return chr(int(digits, 16)), index + 2 + width


def _decode_ansi_c(text: str) -> str:
    """Return `text` with its ANSI-C escapes resolved as Bash resolves them.

    A numeric escape can spell an option letter, so ``$'\\x2dn'`` is the
    option word ``-n`` and has to be read as one.

    Examples
    --------
    >>> _decode_ansi_c(chr(92) + "x2dn")
    '-n'
    >>> _decode_ansi_c(chr(92) + "u006e")
    'n'
    >>> _decode_ansi_c("a" + chr(92) + "nb")
    'a\\nb'
    >>> _decode_ansi_c(chr(92) + "q")
    '\\\\q'
    """
    decoded: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "\\":
            decoded.append(text[index])
            index += 1
            continue
        char, after = _numeric_escape(text, index)
        if char is not None:
            decoded.append(char)
            index = after
            continue
        following = text[index + 1 : index + 2]
        decoded.append(_ANSI_C_SIMPLE.get(following, "\\" + following))
        index += 2
    return "".join(decoded)


def _take_quoted(
    line: str, index: int, *, ansi: bool = False
) -> tuple[str, int]:
    """Return a quoted word's text and the index just past its closing quote.

    The text is the word's content as written, decoded when `ansi` marks a
    ``$'...'`` string; an unterminated quote is not a command the shell runs,
    so it raises rather than reading past the end.

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
    >>> quoted = "'" + chr(92) + "x2dn'"
    >>> _take_quoted(quoted, 0, ansi=True)
    ('-n', 7)
    """
    end = closing_quote(line, index)
    if end is None:
        message = f"unterminated quote in {line!r}"
        raise ValueError(message)
    text = line[index + 1 : end]
    return (_decode_ansi_c(text) if ansi else text), end + 1


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
    >>> mask_quotes("set $" + chr(39) + chr(92) + "x2dn" + chr(39)).quoted
    ['-n']
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
        text, index = _take_quoted(
            line, index, ansi=char == "$"
        )
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
