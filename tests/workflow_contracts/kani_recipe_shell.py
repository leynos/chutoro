"""Read a Kani recipe line as the shell that will run it reads it.

GNU make hands each recipe line to ``/bin/sh``, so what a command selects
depends on the shell's own rules rather than on the text's appearance. This
module answers the questions that need those rules: where a comment begins,
whether a trailing ``|| true`` swallows a failure, which unquoted control
operators chain a second command, and where a quoted run opens or closes.

Re-use policy: import from :mod:`kani_recipe_reading` only. The shell rules are
not make's, so a second consumer should extend this module rather than restate
them beside a different parser.
"""

from __future__ import annotations

import re

#: Recipe text that swallows a failure. GNU make reads a leading ``-`` as
#: ignore-the-status, and a trailing ``|| true`` (or ``|| :``) does the same for
#: the shell. The shell ignores spacing around its tokens, so `||   true` is the
#: same operator; matching a fixed list of spellings would accept it.
_MASKING_SUFFIX = re.compile(r"\|\|\s*(?:true|:)\s*$")

#: Shell control operators this reader refuses. A tier recipe is one command
#: that make can fail on; a `;` lets a later `true` hide the checker's status,
#: and `&&`/`&`/`|` do the same in other shapes. Quoted text is exempt.
_CONTROL_OPERATORS = frozenset({";", "&&", "&", "|"})


def terminal_command(arguments: str) -> str:
    """Return the command text with any trailing shell comment removed.

    A shell comment is not part of the command the shell runs, so nothing
    written after one is a package, a harness selector or an option. The scan
    tracks quotes and escapes, because a hash inside a quoted value is text the
    shell passes on: `--target-dir "build # local"` hides no option from the
    checker, and cutting at the hash would hide any option after it.
    """
    start = comment_start(arguments)
    if start is None:
        return arguments.rstrip()
    return arguments[:start].rstrip()


def comment_start(arguments: str) -> int | None:
    """Return the offset of an unquoted shell comment's ``#``, if any.

    A comment begins at a word boundary, so an escaped space keeps the hash
    that follows it inside the word: a `build` whose space is escaped still
    passes `--jobs=2`, and treating the hash as a comment would hide it.
    """
    quote = ""
    word_start = True
    index = 0
    while index < len(arguments):
        character = arguments[index]
        if quote:
            index, quote = quoted_step(arguments, index, quote)
            word_start = False
            continue
        if character == "\\":
            index += 2
            word_start = False
            continue
        if character in {"'", '"'}:
            quote = character
            word_start = False
        elif character == "#" and word_start:
            return index
        else:
            word_start = character.isspace()
        index += 1
    return None


def quoted_step(text: str, index: int, quote: str) -> tuple[int, str]:
    """Advance one character inside a quoted run, closing it at the quote.

    An escaped character inside double quotes is skipped whole, because the
    shell does not close the quote on it.
    """
    character = text[index]
    if character == "\\" and quote == '"':
        return index + 2, quote
    if character == quote:
        return index + 1, ""
    return index + 1, quote


def masked(command: str) -> bool:
    """Report whether `command` swallows its own failure."""
    return _MASKING_SUFFIX.search(command) is not None


def control_operators(command: str) -> list[str]:
    """Return the unquoted shell control operators in `command`."""
    found: list[str] = []
    quote = ""
    index = 0
    while index < len(command):
        character = command[index]
        if quote or character in {"\\", "'", '"'}:
            index, quote = _operator_step(command, index, quote)
            continue
        pair = command[index : index + 2]
        if pair in {"&&", "||"}:
            found.append(pair)
            index += 2
            continue
        if character in _CONTROL_OPERATORS:
            found.append(character)
        index += 1
    return found


def _operator_step(command: str, index: int, quote: str) -> tuple[int, str]:
    """Advance one character of quoting or escaping in `command`."""
    character = command[index]
    if quote:
        if character == "\\" and quote == '"':
            return index + 2, quote
        return index + 1, "" if character == quote else quote
    if character == "\\":
        # An escaped quote is text, so the operator after it is still one.
        return index + 2, quote
    return index + 1, character