"""Read one shell command for what it runs and what would mask it.

A Makefile recipe line and a CI step are shell command lines, and a line that
mentions nextest is not a line that runs it. ``echo $(CARGO) nextest run``
prints the command, ``nextest run || true`` runs it and reports success
whatever it did, and ``nextest run --no-run`` compiles it without executing a
test. Telling those apart is reading shell syntax: comments end where a word
starts, quoted text is data even where it spells an operator, an operator ends
one simple command and begins another, and the words are split the way the
shell splits them rather than on whitespace.

This module holds the shell-level reading: quote boundaries, comments, and the
words a command is made of. The masking that sets quoted text aside so an
operator inside it is never read as one lives in ``scheduling_quote_reading``,
which this module supplies the quote scan to. The recipe itself, its target,
and the filter expressions it passes live in ``scheduling_reading``, the
CI-step reading lives in ``scheduling_script_reading``, its line joining lives
in ``scheduling_line_reading``, its command segmentation lives in
``scheduling_segment_reading``, the here-documents it drops live in
``scheduling_here_document_reading``, the nextest invocation selected from a
command lives in ``scheduling_invocation_reading``, and the declarations the
recipe names live in ``rust_declaration_reading``.

Re-use policy: import from ``scheduling_reading``,
``scheduling_quote_reading``, ``scheduling_script_reading``,
``scheduling_line_reading``, ``scheduling_segment_reading``,
``scheduling_here_document_reading``, ``scheduling_invocation_reading`` and
``scheduling_contract_test`` only. Those files were pushed past the 400-line
limit ``AGENTS.md`` sets when these readers lived beside the recipe, and this
module keeps each one inside it. A new shell rule belongs here with the shell
behaviour it models stated beside it, so a later reader can check the rule
against the shell rather than against the tests.
"""

from __future__ import annotations

import re
import shlex
import typing as typ

#: Characters after which the shell can start a new word. An operator ends the
#: word before it, so a ``#`` written straight after one is still at a word
#: start.
_OPERATOR_CHARS = frozenset(";&|()<>")

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


def _after_substitution(line: str, start: int) -> int:
    """Return the index just past the ``$(...)`` at `start`.

    A command substitution is parsed as a command of its own, so a quote
    inside it groups its own words and closes nothing that surrounds it.
    Its parentheses nest, and a backslash escapes the character after it. An
    unclosed substitution returns one past the line's end, so a caller that
    scans for a closing quote runs off the same end and finds none.

    Examples
    --------
    >>> _after_substitution('$(printf "%s" " #") x', 0)
    19
    >>> _after_substitution("$(unclosed", 0) == len("$(unclosed")
    True
    >>> adjacent = "$(printf " + "'a''b'" + ")"
    >>> _after_substitution(adjacent, 0) == len(adjacent)
    True
    """
    depth = 1
    index = start + 2
    while index < len(line):
        grouped = _after_grouped_text(line, index)
        if grouped != index:
            index = grouped
            continue
        depth += _grouping_delta(line[index])
        if depth == 0:
            return index + 1
        index += 1
    return len(line)


def _grouping_delta(char: str) -> int:
    """Return how `char` changes the parenthesis depth.

    Examples
    --------
    >>> _grouping_delta("(")
    1
    >>> _grouping_delta(")")
    -1
    >>> _grouping_delta("a")
    0
    """
    if char == "(":
        return 1
    return -1 if char == ")" else 0


def _after_grouped_text(line: str, index: int) -> int:
    """Return the index of the text that decides how `line[index]` groups.

    A backslash and the character after it are one escaped character, and a
    quote opens text that closes at its own quote, so neither contributes a
    parenthesis. An unclosed quote runs to the end of the line, where the
    caller's loop ends too.

    Examples
    --------
    >>> _after_grouped_text("a(b", 1)
    1
    >>> escaped = chr(92) + "(b"
    >>> _after_grouped_text(escaped, 0)
    2
    >>> _after_grouped_text('"(" b', 0)
    3
    >>> _after_grouped_text("'unclosed", 0) == len("'unclosed")
    True
    """
    char = line[index : index + 1]
    if char == "\\":
        return index + 2
    if char in {"'", '"'}:
        end = closing_quote(line, index)
        return len(line) if end is None else end + 1
    return index


def closing_quote(line: str, start: int) -> int | None:
    """Return the index of the quote that closes the one at `start`.

    A double-quoted string and a Bash ANSI-C string (``$'...'``) may escape
    their own quote, so a backslash is skipped together with the character
    after it. A command substitution inside a double-quoted string is parsed
    as its own command, so its quotes do not close the string around it. An
    ordinary single-quoted string has no escapes. An unclosed quote returns
    ``None``, because whether that is an error depends on what the caller is
    reading.

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
    >>> substituted = '"$(printf "%s" " #")"'
    >>> closing_quote(substituted, 0) == len(substituted) - 1
    True
    >>> closing_quote("$'$('", 1)
    4
    """
    quote = line[start]
    escape_aware = quote == '"' or _is_ansi_c_string(line, start)
    index = start + 1
    while index < len(line):
        if escape_aware and line[index] == "\\":
            index += 2
            continue
        if quote == '"' and line.startswith("$(", index):
            index = _after_substitution(line, index)
            continue
        if line[index] == quote:
            return index
        index += 1
    return None


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


#: Redirection operators, with the word after each being a file name rather
#: than an argument. A leading file-descriptor number is part of the operator,
#: as in ``2>``.
_REDIRECTION = re.compile(r"\d*(?:&>>?|>>|>|<|<>|>\||<<<|<<)")


def is_redirection(token: str) -> bool:
    """Return whether `token` is a redirection operator.

    ``2>&1`` arrives as three tokens: the operator ``2>``, the ``&`` that
    copies a descriptor and the descriptor ``1``. Reading the operator is what
    keeps that ``&`` from being taken for a background operator. An attached
    target such as ``>scheduling#log`` is one word rather than an operator,
    and a word that merely ends in ``>`` is not one either.

    Examples
    --------
    >>> is_redirection(">")
    True
    >>> is_redirection("2>")
    True
    >>> is_redirection(">scheduling#log")
    False
    >>> is_redirection("scheduling>")
    False
    """
    return _REDIRECTION.fullmatch(token) is not None


def starts_a_redirection(token: str) -> bool:
    """Return whether `token` begins with a redirection operator.

    ``>/dev/null`` is one word: the operator and its target are attached. A
    caller that strips redirections drops such a word whole.

    Examples
    --------
    >>> starts_a_redirection(">/dev/null")
    True
    >>> starts_a_redirection("2>&1")
    True
    >>> starts_a_redirection("scheduling>")
    False
    >>> starts_a_redirection("'>/dev/null'")
    False
    """
    return _REDIRECTION.match(token) is not None
