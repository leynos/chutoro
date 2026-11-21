"""Read one shell command for what it runs and what would mask it.

A Makefile recipe line and a CI step are shell command lines, and a line that
mentions nextest is not a line that runs it. ``echo $(CARGO) nextest run``
prints the command, ``nextest run || true`` runs it and reports success
whatever it did, and ``nextest run --no-run`` compiles it without executing a
test. Telling those apart is reading shell syntax: comments end where a word
starts, quoted text is data even where it spells an operator, an operator ends
one simple command and begins another, and the words are split the way the
shell splits them rather than on whitespace.

This module holds the shell-level reading: quotes, comments, the words a
command is made of, and the masking that sets quoted text aside so an operator
inside it is never read as one. The recipe itself, its target, and the filter
expressions it passes live in ``scheduling_reading``, the CI-step reading
lives in ``scheduling_script_reading``, its line joining lives in
``scheduling_line_reading``, its command segmentation lives in
``scheduling_segment_reading``, the here-documents it drops live in
``scheduling_here_document_reading``, the nextest invocation selected from a
command lives in ``scheduling_invocation_reading``, and the declarations the
recipe names live in ``rust_declaration_reading``.

Re-use policy: import from ``scheduling_reading``,
``scheduling_script_reading``, ``scheduling_line_reading``,
``scheduling_segment_reading``, ``scheduling_here_document_reading``,
``scheduling_invocation_reading`` and ``scheduling_contract_test`` only. Those
files were pushed past the 400-line
limit ``AGENTS.md`` sets when these readers lived beside the recipe, and this
module keeps each one inside it. A new shell rule belongs here with the shell
behaviour it models stated beside it, so a later reader can check the rule
against the shell rather than against the tests.
"""

from __future__ import annotations

import shlex
import typing as typ

#: Characters after which the shell can start a new word. An operator ends the
#: word before it, so a ``#`` written straight after one is still at a word
#: start.
_OPERATOR_CHARS = frozenset(";&|()<>")

#: The characters that separate words for the shell.
_WORD_SEPARATORS = frozenset(" \t\n")


def _is_ansi_c_string(line: str, start: int) -> bool:
    """Return whether the quote at `start` opens a Bash ``$'...'`` string.

    Examples
    --------
    >>> _is_ansi_c_string("$'a'", 1)
    True
    >>> _is_ansi_c_string("'a'", 0)
    False
    """
    return start > 0 and line[start - 1] == "$"


def closing_quote(line: str, start: int) -> int | None:
    """Return the index of the quote that closes the one at `start`.

    A double-quoted string and a Bash ANSI-C string (``$'...'``) may escape
    their own quote, so a backslash is skipped together with the character
    after it. An ordinary single-quoted string has no escapes. An unclosed
    quote returns ``None``, because whether that is an error depends on what
    the caller is reading.

    Parameters
    ----------
    line
        The command line containing the quote.
    start
        The index of the opening quote.

    Returns
    -------
    int | None
        The index of the closing quote, or ``None`` when the quote is never
        closed. Callers that cannot read the rest of the line raise
        ``ValueError`` for that case.

    Examples
    --------
    >>> closing_quote("'a b' c", 0)
    4
    >>> closing_quote("'unclosed", 0) is None
    True
    >>> escaped = '"a ' + chr(92) + '" b"'
    >>> closing_quote(escaped, 0)
    7
    >>> ansi = "$'a" + chr(92) + "'b'"
    >>> closing_quote(ansi, 1)
    6
    """
    quote = line[start]
    escape_aware = quote == '"' or _is_ansi_c_string(line, start)
    index = start + 1
    while index < len(line):
        if line[index] == "\\" and escape_aware:
            index += 2
            continue
        if line[index] == quote:
            return index
        index += 1
    return None


#: The form a quoted word is set aside in while a line's shell syntax is read.
#: The marker holds no operator or comment character, so masking the text keeps
#: it whole through every later reading.
_QUOTED = "\x01{}\x02"


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


def _after_quoted(line: str, start: int) -> int:
    """Return the index just past the quoted text that opens at `start`.

    An unterminated quote is not a command the shell runs, so reading past it
    would silently invent a command line, and it raises instead.

    Examples
    --------
    >>> _after_quoted("'a b' c", 0)
    5
    >>> _after_quoted("'unclosed", 0)
    Traceback (most recent call last):
        ...
    ValueError: unterminated quote in "'unclosed"
    """
    end = closing_quote(line, start)
    if end is None:
        message = f"unterminated quote in {line!r}"
        raise ValueError(message)
    return end + 1


def _is_word_start(line: str, index: int) -> bool:
    """Return whether the ``#`` at `index` begins a word the shell can see.

    ``shlex`` tokenizes with shell separators, and it silently stops a word
    only when a quote is left open; every other token includes the characters
    between it and the next separator. A ``#`` is therefore a comment only when
    it starts a word: at the beginning, after whitespace — which a redirection
    written ``>scheduling#log`` does not have — or right after an operator,
    which ends the word a ``#`` can start a new one at.

    Examples
    --------
    >>> _is_word_start("make test # run", 10)
    True
    >>> _is_word_start("make test#run", 9)
    False
    >>> _is_word_start("# run", 0)
    True
    >>> _is_word_start("run >log;# note", 9)
    True
    """
    if index == 0:
        return True
    previous = line[index - 1]
    return previous in _WORD_SEPARATORS or previous in _OPERATOR_CHARS


def _starts_comment(line: str, index: int, *, escaped: bool) -> bool:
    """Return whether the ``#`` at `index` begins a shell comment.

    A comment starts only at a word start, and a backslash in front of the
    ``#`` makes it a literal character of the word it sits in.

    Examples
    --------
    >>> _starts_comment("run # note", 4, escaped=False)
    True
    >>> _starts_comment("run#note", 3, escaped=False)
    False
    >>> _starts_comment("run #note", 4, escaped=True)
    False
    """
    if escaped:
        return False
    return _is_word_start(line, index)


def _comment_start(line: str) -> int | None:
    """Return the index of the first shell comment in `line`, if any.

    The scan is quote-aware because a ``#`` inside quoted text is data. It is
    also escape-aware, because a backslash makes the next character literal: in
    ``>scheduling\\;#log`` the escaped ``;`` ends nothing, so the ``#`` sits
    inside the word and the ``|| true`` after it still masks a failure. An
    unterminated quote is not a command the shell runs, so it raises rather
    than reading the rest of the line as an argument.

    Examples
    --------
    >>> _comment_start("cargo nextest run -E 'a # b'")
    >>> _comment_start("cargo nextest run # --no-tests=fail")
    18
    >>> _comment_start("cargo nextest run >scheduling\\\\;#log || true")
    """
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\":
            index += 2
            escaped = True
            continue
        if char in "'\"":
            index = _after_quoted(line, index)
            escaped = False
            continue
        if char == "#" and _starts_comment(line, index, escaped=escaped):
            return index
        escaped = False
        index += 1
    return None


def strip_shell_comment(line: str) -> str:
    """Return `line` without its trailing shell comment.

    A ``#`` starts a comment only where a word starts, so it must begin the
    line, follow whitespace, or follow an operator, and a ``#`` inside a word
    is part of that word. Quoted text is data even where a comment could start.
    ``cargo nextest run >scheduling#log || true`` therefore runs nextest and
    masks its failure: cutting at the ``#`` left a command that looked unmasked
    and passed the invocation check. Comments are removed before any assertion
    reads a command, because a comment that mentioned a flag would otherwise
    stand in for the flag on an invocation that omits it.

    Parameters
    ----------
    line
        The command line to read.

    Returns
    -------
    str
        The line with any trailing shell comment removed.

    Examples
    --------
    >>> strip_shell_comment("make test # run the suite")
    'make test'
    >>> strip_shell_comment("cargo nextest run >log#1 || true")
    'cargo nextest run >log#1 || true'
    >>> strip_shell_comment("nextest run -E 'a # b'")
    "nextest run -E 'a # b'"
    >>> strip_shell_comment("make test")
    'make test'
    """
    comment = _comment_start(line)
    return line if comment is None else line[:comment].rstrip()


def shell_tokens(command: str) -> list[str]:
    """Split `command` into words, keeping quoted text whole.

    ``shlex.split`` leaves an unspaced ``||true`` attached to the word before
    it, so a separator check on the resulting tokens misses it. Splitting on
    the separator characters themselves still respects quotes, and leaves a
    word such as ``$(CARGO)`` intact because only separators are punctuation.

    ``shlex`` treats ``#`` as a comment character of its own and silently stops
    at one, which would drop a redirection such as ``>scheduling#log`` and every
    mask after it. Comments are already removed, with shell word boundaries, by
    ``strip_shell_comment``; the lexer must not remove them a second time.

    Parameters
    ----------
    command
        The command to split.

    Returns
    -------
    list[str]
        The command's words, with quoted text kept whole.

    Examples
    --------
    >>> shell_tokens("$(CARGO) nextest run ||true")
    ['$(CARGO)', 'nextest', 'run', '||', 'true']
    >>> shell_tokens("nextest run -E 'test(=a) | all()'")
    ['nextest', 'run', '-E', 'test(=a) | all()']
    >>> shell_tokens("nextest run >log#1 || true")
    ['nextest', 'run', '>log#1', '||', 'true']
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.commenters = ""
    lexer.whitespace_split = True
    return list(lexer)

