"""Read a shell command's comments and the words they leave behind.

A shell comment starts at an unescaped ``#`` that begins a word, and it runs to
the end of the physical line; a ``#`` inside a word, as in a redirection to
``log#1``, is part of that word. A command's comments are removed before its
words are read, because the lexer would otherwise stop at a ``#`` it cannot
distinguish from data.

This module holds that reading; ``scheduling_command_reading`` supplies the
quote scan it relies on, and the split keeps both files inside the 400-line
limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_command_reading``,
``scheduling_line_reading`` and ``scheduling_reading``, which read command
lines. A new comment rule belongs here with the shell behaviour it models
stated beside it.
"""

from __future__ import annotations

from scheduling_command_reading import closing_quote

#: The characters that separate words for the shell.
_WORD_SEPARATORS = frozenset(" \t\n")

#: Characters after which the shell can start a new word. An operator ends the
#: word before it, so a ``#`` written straight after one is still at a word
#: start.
_OPERATOR_CHARS = frozenset(";&|()<>")


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
