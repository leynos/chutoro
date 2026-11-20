"""Read one shell command for what it runs and what would mask it.

A Makefile recipe line and a CI step are shell command lines, and a line that
mentions nextest is not a line that runs it. ``echo $(CARGO) nextest run``
prints the command, ``nextest run || true`` runs it and reports success
whatever it did, and ``nextest run --no-run`` compiles it without executing a
test. Telling those apart is reading shell syntax: comments end where a word
starts, an operator ends one simple command and begins another, and the words
are split the way the shell splits them rather than on whitespace.

This module holds that reading for the scheduling gate. The recipe itself, its
target, and the filter expressions it passes live in ``scheduling_reading``,
which imports these readers; the declarations the recipe names live in
``rust_declaration_reading``.

Re-use policy: import from ``scheduling_reading`` and
``scheduling_contract_test`` only. Both files were pushed past the 400-line
limit ``AGENTS.md`` sets when these readers lived beside the recipe, and this
module keeps each one inside it. A new shell rule belongs here with the shell
behaviour it models stated beside it, so a later reader can check the rule
against the shell rather than against the tests.
"""

from __future__ import annotations

import shlex

#: The spellings of the Cargo executable a scheduling invocation may start
#: with. `$(CARGO)` is what the Makefile uses; a literal `cargo` is accepted
#: because it runs the same command. A command that only mentions nextest in
#: an argument, such as `echo $(CARGO) nextest run ...`, runs nothing, and
#: every assertion about its arguments would still pass.
CARGO_COMMANDS = ("$(CARGO)", "cargo")

#: Shell separators that continue a recipe line after the nextest invocation.
#: `|| true` and `; true` make the line succeed after nextest fails, so the
#: target stays green while the regression it exists to run is broken.
SHELL_SEPARATORS = frozenset({";", ";;", "&&", "||", "|", "&"})

#: Flags that compile without running. `--no-run` builds the tests and executes
#: none of them, so an invocation carrying it produces no runtime evidence even
#: though its selector, count and flags all look right.
COMPILE_ONLY_FLAGS = ("--no-run",)

#: Characters after which the shell can start a new word. An operator ends the
#: word before it, so a ``#`` written straight after one is still at a word
#: start.
_OPERATOR_CHARS = frozenset(";&|()<>")

#: The characters that separate words for the shell.
_WORD_SEPARATORS = frozenset(" \t\n")


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
    quote = ""
    previous_was_escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            if char == "\\" and quote == '"':
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char == "\\":
            index += 2
            previous_was_escaped = True
            continue
        if char in "'\"":
            quote = char
        elif (
            char == "#"
            and not previous_was_escaped
            and _is_word_start(line, index)
        ):
            return index
        previous_was_escaped = False
        index += 1
    if quote:
        message = f"unterminated quote in {line!r}"
        raise ValueError(message)
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


def is_nextest_invocation(command: str) -> bool:
    """Return whether `command` runs nextest in its executable position.

    A recipe line that only mentions nextest in an argument runs nothing.
    Three other shapes run it and still leave the target green when the
    regression fails: Make's ignore-error prefix, `|| true`, and a second
    command after `;`. A fourth runs nothing at all: `--no-run` compiles the
    tests without executing them, so the selector is satisfied and the
    evidence is not produced. Each spelling that is not the plain Cargo
    command leaves the line out of the invocation list, where the
    one-per-regression count rejects it.

    Examples
    --------
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core --lib")
    True
    >>> is_nextest_invocation("echo $(CARGO) nextest run -p chutoro-core")
    False
    >>> is_nextest_invocation("-$(CARGO) nextest run -p chutoro-core")
    False
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core || true")
    False
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core ||true")
    False
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core --no-run")
    False
    """
    tokens = shell_tokens(command)
    rejected = SHELL_SEPARATORS.union(COMPILE_ONLY_FLAGS)
    if any(token in rejected for token in tokens):
        return False
    return (
        len(tokens) >= 3
        and tokens[0] in CARGO_COMMANDS
        and tokens[1:3] == ["nextest", "run"]
    )