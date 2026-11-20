"""Read one shell command for what it runs and what would mask it.

A Makefile recipe line and a CI step are shell command lines, and a line that
mentions nextest is not a line that runs it. ``echo $(CARGO) nextest run``
prints the command, ``nextest run || true`` runs it and reports success
whatever it did, and ``nextest run --no-run`` compiles it without executing a
test. Telling those apart is reading shell syntax: comments end where a word
starts, quoted text is data even where it spells an operator, an operator ends
one simple command and begins another, and the words are split the way the
shell splits them rather than on whitespace.

This module holds that reading for the scheduling gate. The recipe itself, its
target, and the filter expressions it passes live in ``scheduling_reading``,
the CI-step reading lives in ``scheduling_script_reading``, its line joining
and quote masking live in ``scheduling_line_reading``, and the declarations
the recipe names live in ``rust_declaration_reading``.

Re-use policy: import from ``scheduling_reading``,
``scheduling_script_reading``, ``scheduling_line_reading`` and
``scheduling_contract_test`` only. Those files were pushed past the 400-line
limit ``AGENTS.md`` sets when these readers lived beside the recipe, and this
module keeps each one inside it. A new shell rule belongs here with the shell
behaviour it models stated beside it, so a later reader can check the rule
against the shell rather than against the tests.
"""

from __future__ import annotations

import re
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


#: Redirection operators, with Cargo's word after each being a file name
#: rather than an argument. A leading file-descriptor number is part of the
#: operator, as in ``2>``.
_REDIRECTION = re.compile(r"\d*(?:&>>?|>>|>|<|<>|>\||<<<|<<)")


def _is_redirection(token: str) -> bool:
    """Return whether `token` is a redirection operator.

    Examples
    --------
    >>> _is_redirection(">")
    True
    >>> _is_redirection("2>")
    True
    >>> _is_redirection(">log#1")
    False
    """
    return _REDIRECTION.fullmatch(token) is not None


def command_arguments(command: str) -> list[str]:
    """Return `command`'s words with its redirections and their targets removed.

    A redirection's target is a file name, not an argument. In
    ``nextest run --no-tests=pass > --no-tests=fail`` the only
    ``--no-tests=fail`` is the file written to, and the command still runs
    with ``--no-tests=pass``, so a check that matched the whole word list
    would accept the opposite setting. An attached form such as ``>log#1`` is
    one word and can never be read as a flag, but the spaced form arrives as
    an operator and a separate word, which is why the target is dropped here
    rather than matched as a word.

    Examples
    --------
    >>> command_arguments("nextest run --no-tests=pass > --no-tests=fail")
    ['nextest', 'run', '--no-tests=pass']
    >>> command_arguments("nextest run --no-tests=fail >log#1")
    ['nextest', 'run', '--no-tests=fail', '>log#1']
    >>> command_arguments("nextest run 2> --no-tests=fail --lib")
    ['nextest', 'run', '--lib']
    """
    arguments: list[str] = []
    redirects_next = False
    for token in shell_tokens(command):
        if redirects_next:
            redirects_next = False
            continue
        if _is_redirection(token):
            redirects_next = True
            continue
        arguments.append(token)
    return arguments


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