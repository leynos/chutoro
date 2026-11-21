"""Split a step's command lines into the simple commands the shell runs.

The scheduling contract asks whether a step's script executes the nextest
installer, and the answer is about commands rather than text. One physical
line can hold several commands — ``a ; b``, ``a && b``, ``a | b`` — and each
of them carries the operator that follows it, because that operator decides
what the line reports when the command fails. Reading the line into (words,
operator) segments is what makes ``installer || true`` a command whose failure
is masked rather than a word beginning with the installer path.

This module holds that segmentation alone, from an already-joined line and
from a whole script. ``scheduling_script_reading`` decides which segment is a
blocking installation, and the split keeps each file inside the 400-line limit
``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_script_reading`` only; this module
reuses the quote masking and tokenizing in ``scheduling_command_reading`` and
the line joining in ``scheduling_line_reading``. A new segmentation rule
belongs here with the shell behaviour it models stated beside it.
"""

from __future__ import annotations

import typing as typ

from scheduling_command_reading import mask_quotes, restore_quotes, shell_tokens
from scheduling_line_reading import logical_lines

#: Characters that end one simple command and begin another. ``shlex`` yields
#: any run of them as one token, so ``|&`` arrives whole.
_OPERATOR_CHARS = frozenset(";&|")


class Segment(typ.NamedTuple):
    """One simple command and the operator that ends it.

    The operator is kept because it decides what the job sees when the
    command fails: ``installer || true`` runs the installer and reports
    success, so a segment read without its operator cannot say whether the
    installation was allowed to fail.
    """

    words: list[str]
    follows: str


def read_segments(tokens: list[str], quoted: list[str]) -> list[Segment]:
    """Return one line's simple commands, each with the operator that ends it.

    ``shlex`` yields any run of operator characters as one token, so ``|&``
    arrives whole and an unspaced ``||true`` still separates. A word's quoted
    text was set aside before tokenizing and is restored here, so a command is
    read with the words the shell would give it.

    Examples
    --------
    >>> read_segments(["a", "&&", "b"], [])
    [Segment(words=['a'], follows='&&'), Segment(words=['b'], follows='')]
    >>> read_segments(["a", "|&", "b"], [])
    [Segment(words=['a'], follows='|&'), Segment(words=['b'], follows='')]
    """
    segments: list[Segment] = []
    words: list[str] = []
    for token in tokens:
        if set(token) <= _OPERATOR_CHARS:
            if words:
                segments.append(Segment(words, token))
            words = []
            continue
        words.append(restore_quotes(token, quoted))
    if words:
        segments.append(Segment(words, ""))
    return segments


def read_lines(script: str) -> list[list[Segment]]:
    """Return each line of `script` as the simple commands it runs.

    A step's script can hold several commands, separated by newlines or by
    shell operators, and a comment line names a script without running it.
    Splitting first is what makes "is the installer executed?" a question
    about a command's first word rather than about a substring. The operator
    that follows each command is retained, because an operator decides what
    the line reports when the command fails. Segments are grouped by line
    because a command's status travels only as far as the end of its own line.

    Lines arrive with their continuations resolved and their comments removed
    by ``scheduling_line_reading``, so a ``#`` inside a redirection stays part
    of its word. A here-document's body never arrives: the module that joins
    the lines drops it, because the body is stdin for the command that opened
    the document rather than commands of its own. Each line's quoted text is
    masked, so a quoted operator stays an argument, and the lexer's own
    comment handling is turned off so it cannot cut the line a second time.

    Examples
    --------
    >>> read_lines("echo scripts/install-nextest.sh")
    [[Segment(words=['echo', 'scripts/install-nextest.sh'], follows='')]]
    >>> [segment.words for segment in read_lines("a || true")[0]]
    [['a'], ['true']]
    >>> [segment.follows for segment in read_lines("a || true")[0]]
    ['||', '']
    >>> read_lines("# scripts/install-nextest.sh")
    [[]]
    >>> [segment.words for segment in read_lines("echo '&&' installer")[0]]
    [['echo', '&&', 'installer']]
    >>> [segment.words for segment in read_lines("installer >log#1 || true")[0]]
    [['installer', '>log#1'], ['true']]
    >>> [segment.words for segment in read_lines("installer # don't")[0]]
    [['installer']]
    >>> [segment.words for segment in read_lines("true ||" + chr(10) + "x")[0]]
    [['true'], ['x']]
    >>> [segment.follows for segment in read_lines("true ||" + chr(10) + "x")[0]]
    ['||', '']
    >>> escaped = "echo " + chr(92) + "&" + chr(92) + "& installer"
    >>> [segment.words for segment in read_lines(escaped)[0]]
    [['echo', '&&', 'installer']]
    >>> len(read_lines(escaped)[0])
    1
    """
    lines: list[list[Segment]] = []
    for source_line in logical_lines(script):
        masked = mask_quotes(source_line)
        tokens = shell_tokens(masked.text)
        lines.append(read_segments(tokens, masked.quoted))
    return lines