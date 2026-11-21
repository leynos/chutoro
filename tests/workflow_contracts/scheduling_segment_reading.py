"""Split a step's command lines into the simple commands the shell runs.

The scheduling contract asks whether a step's script executes the nextest
installer, and the answer is about commands rather than text. One physical
line can hold several commands — ``a ; b``, ``a && b``, ``a | b`` — and each
of them carries the operator that follows it, because that operator decides
what the line reports when the command fails. Reading the line into (words,
operator) segments is what makes ``installer || true`` a command whose failure
is masked rather than a word beginning with the installer path.

This module holds that segmentation, from an already-joined line and from a
whole script, together with the keywords that open a compound command the
simple reading cannot place. ``scheduling_script_reading`` decides which
segment is a blocking installation, and the split keeps each file inside the
400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_script_reading`` only; this module
reuses the tokenizing in ``scheduling_command_reading``, the quote masking in
``scheduling_quote_reading`` and the line joining in
``scheduling_line_reading``. A new segmentation rule belongs here with the
shell behaviour it models stated beside it.
"""

from __future__ import annotations

import typing as typ

from scheduling_command_reading import shell_tokens
from scheduling_quote_reading import mask_quotes, restore_quotes
from scheduling_line_reading import logical_lines

#: Characters that end one simple command and begin another. ``shlex`` yields
#: any run of them as one token, so ``|&`` arrives whole.
_OPERATOR_CHARS = frozenset(";&|")

#: The parentheses that group commands. ``shlex`` leaves them attached to the
#: word beside them, so ``(true`` and ``true)`` arrive as single tokens.
_GROUPING_CHARS = frozenset("()")


class Segment(typ.NamedTuple):
    """One simple command and the operator that ends it.

    The operator is kept because it decides what the job sees when the
    command fails: ``installer || true`` runs the installer and reports
    success, so a segment read without its operator cannot say whether the
    installation was allowed to fail.
    """

    words: list[str]
    follows: str


def _joins_descriptors(words: list[str], token: str) -> bool:
    """Return whether `token` is the ``&`` of a descriptor duplication.

    ``shlex`` splits ``2>&1`` into ``2>``, ``&`` and ``1``. The ``&`` copies
    one descriptor onto another rather than backgrounding the command, so it
    belongs to the redirection rather than to the segment operator.

    Examples
    --------
    >>> _joins_descriptors(["installer", "2>"], "&")
    True
    >>> _joins_descriptors(["installer"], "&")
    False
    >>> _joins_descriptors(["installer"], ";")
    False
    """
    if token != "&" or not words:
        return False
    return words[-1].endswith((">", "<"))


def _flush_segment(
    segments: list[Segment], words: list[str], operator: str
) -> list[str]:
    """Append the command `words` hold, ending it with `operator`, if any.

    The returned list is the empty word list the next command starts from.

    Examples
    --------
    >>> segments = []
    >>> _flush_segment(segments, ["a"], "&&")
    []
    >>> segments
    [Segment(words=['a'], follows='&&')]
    >>> _flush_segment(segments, [], ";")
    []
    >>> segments
    [Segment(words=['a'], follows='&&')]
    """
    if words:
        segments.append(Segment(words, operator))
    return []


def read_segments(tokens: list[str], quoted: list[str]) -> list[Segment]:
    """Return one line's simple commands, each with the operator that ends it.

    ``shlex`` yields any run of operator characters as one token, so ``|&``
    arrives whole and an unspaced ``||true`` still separates. A word's quoted
    text was set aside before tokenizing and is restored here, so a command is
    read with the words the shell would give it. A grouping parenthesis may
    arrive attached to a word, as ``(true``, and no simple reading can place
    the group it opens; the line is reported as one such segment instead of
    being read as the commands it merely contains. A quoted or escaped
    parenthesis was masked before tokenizing and stays an ordinary word.

    Examples
    --------
    >>> read_segments(["a", "&&", "b"], [])
    [Segment(words=['a'], follows='&&'), Segment(words=['b'], follows='')]
    >>> read_segments(["a", "|&", "b"], [])
    [Segment(words=['a'], follows='|&'), Segment(words=['b'], follows='')]
    >>> read_segments(["(true"], [])
    [Segment(words=['('], follows='')]
    >>> read_segments(["(", "a", ")"], [])
    [Segment(words=['('], follows='')]
    >>> read_segments(["installer", "2>", "&", "1"], [])
    [Segment(words=['installer', '2>', '&', '1'], follows='')]
    """
    if any(_GROUPING_CHARS & set(token) for token in tokens):
        return [Segment(["("], "")]
    segments: list[Segment] = []
    words: list[str] = []
    for token in tokens:
        if _joins_descriptors(words, token):
            words.append("&")
            continue
        if set(token) <= _OPERATOR_CHARS:
            words = _flush_segment(segments, words, token)
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

#: Words that open or close a compound command whose body this reading cannot
#: place. ``if false; then``, ``fi`` and the loop and case keywords are
#: commands of their own to a simple reading, so an installer written inside
#: the body reads exactly like a plain command that certainly ran, while the
#: shell skips the whole body. A parenthesised group is the same: Bash parses
#: it as one command, so ``false && (`` makes every line up to ``)`` a body
#: that may never run. The reading tracks no block structure, so a script
#: holding one of these yields no evidence at all rather than a guess.
COMPOUND_KEYWORDS = frozenset(
    {
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "select",
        "function",
        "{",
        "}",
        "(",
        ")",
        "coproc",
    }
)


def opens_a_compound_command(segment: Segment) -> bool:
    """Return whether `segment` is a keyword this reading cannot place.

    A keyword opens or closes a compound command only in command position, so
    ``echo if`` prints the word rather than opening a branch.

    Examples
    --------
    >>> opens_a_compound_command(Segment(["if", "false"], ";"))
    True
    >>> opens_a_compound_command(Segment(["echo", "if"], ""))
    False
    """
    return bool(segment.words) and segment.words[0] in COMPOUND_KEYWORDS


def has_a_compound_command(lines: list[list[Segment]]) -> bool:
    """Return whether `lines` holds a keyword this reading cannot place.

    A compound command's keywords stand as commands of their own to a simple
    reading, so without this the installer inside an untaken body would count
    as one that certainly ran. The reading has no block structure to say
    which branch was taken, so the whole script is refused instead.

    Examples
    --------
    >>> has_a_compound_command(read_lines("if false; then" + chr(10) + "fi"))
    True
    >>> has_a_compound_command(
    ...     read_lines("for job in a b; do" + chr(10) + "done")
    ... )
    True
    >>> has_a_compound_command(read_lines("true" + chr(10) + "installer"))
    False
    >>> has_a_compound_command(read_lines("echo if"))
    False
    """
    return any(
        opens_a_compound_command(segment)
        for line in lines
        for segment in line
    )
