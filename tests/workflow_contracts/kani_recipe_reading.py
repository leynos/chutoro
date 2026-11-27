"""Read the Makefile's Kani tiers as the commands they execute.

`make kani` is the pull-request gate and `make kani-full` is the nightly tier,
so what each means depends on exactly which `cargo kani` invocations its recipe
runs. This module turns a recipe into those invocations: the package each one
selects, the harness selectors it passes, and the checker wrapper around it.

The reading is deliberately narrow. Only a tab-indented line is executable
recipe text, so a `--harness` in a comment, echoed, or printed by `printf` is
not a selection. A line that masks its own failure — a leading `-` or a
trailing `|| true` — is rejected rather than counted, because a tier whose
recipes cannot fail is not a gate.

Re-use policy: import from :mod:`kani_fast_tier_test` and
:mod:`kani_tier_routing_test` only. A new Makefile reading belongs here beside
the make behaviour it models.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
from kani_recipe_shell import control_operators, masked, terminal_command
from workflow_support import ROOT

MAKEFILE_PATH = ROOT / "Makefile"

#: The Kani tiers and the checker build live in an included fragment, so the
#: prover surface has one owner. The contracts read that file for recipes.
KANI_MAKEFILE_PATH = ROOT / "tools" / "kani" / "kani.mk"

#: The ``kani:`` target header and its immediately following recipe lines. The
#: capture stops at the first line that belongs to neither the recipe nor a
#: comment, and deliberately does not stop at a blank line or a column-0 ``#``
#: line: make permits a blank line between recipe lines, a tab-indented comment
#: is recipe text the shell runs, and a column-0 comment is a make comment that
#: does not end the recipe. Ending the capture at any of them would silently
#: read a shorter tier, and a failure-ignoring command after the comment would
#: go unchecked.
KANI_TARGET_RE = re.compile(
    r"^kani:(?P<header>[^\n]*)\n(?P<recipe>(?:(?:\t[^\n]*|#[^\n]*)?(?:\n|$))*)",
    re.MULTILINE,
)

#: The same shape for the ``kani-full:`` target.
KANI_FULL_TARGET_RE = re.compile(
    r"^kani-full:(?P<header>[^\n]*)\n(?P<recipe>(?:(?:\t[^\n]*|#[^\n]*)?(?:\n|$))*)",
    re.MULTILINE,
)

#: An executable Kani command in a tier recipe. The optional make-variable
#: prefixes cover the environment and the checker wrapper; a ``#`` comment, an
#: ``echo``, or a ``printf`` cannot satisfy the pattern.
#:
#: The leading flags are make's recipe prefixes, each of which changes how the
#: line runs. ``-`` tells make to ignore the failure and is captured so the
#: caller can reject it. ``@`` runs the line without echoing it, and ``+`` runs
#: it even under ``make -n``; neither changes whether the command runs, so both
#: are consumed here rather than refused.
KANI_COMMAND_RE = re.compile(
    r"^\t(?P<prefixes>[-@+]*)(?:\$\([A-Z_]+\)\s+)*"
    r"(?:\$\(KANI_SUITE\)\s+.*?\s--\s+)?"
    r"(?:\$\([A-Z_]+\)\s+)*\$\(CARGO\)\s+kani(?=\s|$)(?P<arguments>.*)$"
)

#: The checker wrapper around one delegated command: the manifest, the
#: package and the scope, in that order, then the argument separator. The
#: manifest's value is quoted in the recipes so a path containing a space
#: reaches the checker as one argument, so the quotes are optional here.
CHECKER_WRAPPER_RE = re.compile(
    r"^\t[-@+]*\$\(KANI_SUITE\)\s+--manifest\s+\"?\$\(KANI_MANIFEST\)\"?\s+"
    r"--package\s+(?P<package>[A-Za-z0-9_-]+)\s+"
    r"--scope\s+(?P<scope>fast|full)\s+--\s+"
)

class RecipeError(ValueError):
    """Raised when a tier recipe cannot be read as a set of Kani selections."""


def recipe(text: str, pattern: re.Pattern[str], target: str) -> str:
    """Return the recipe body of one target, or raise.

    A command written after a semicolon on the target line is recipe text make
    executes, and both patterns would consume it as header. It is refused
    rather than silently omitted from the checks below.
    """
    match = pattern.search(text)
    if match is None or not match["recipe"]:
        raise RecipeError(f"the Makefile must define a {target}: target with a recipe")
    inline = control_operators(match["header"])
    if inline:
        raise RecipeError(
            f"the {target}: target must declare no command on its header line; "
            f"found {sorted(set(inline))!r} in {match['header'].strip()!r}"
        )
    return match["recipe"].rstrip("\n")


def logical_lines(text: str) -> list[str]:
    """Return one recipe's commands with backslash continuations joined.

    Make passes a recipe line's backslash-newline through to the shell and
    removes the recipe-prefix tab that begins the continuation. The shell then
    removes the pair as a line continuation outside single quotes, so the two
    halves are one word; inside single quotes the shell keeps the pair
    literally, and so does this reader.
    """
    lines = text.splitlines()
    logical: list[str] = []
    index = 0
    while index < len(lines):
        line, index = _joined(lines, index)
        logical.append(line)
    return logical


def _joined(lines: list[str], index: int) -> tuple[str, int]:
    """Return one logical line and the index of the line after it."""
    current = lines[index]
    index += 1
    while current.endswith("\\") and index < len(lines):
        following = lines[index][1:] if lines[index].startswith("\t") else lines[index]
        if _open_single_quote(current):
            current = f"{current}\n{following}"
        else:
            current = f"{current[:-1]}{following}"
        index += 1
    return current, index


def _open_single_quote(text: str) -> bool:
    """Return whether `text` ends inside a single-quoted run."""
    quote = ""
    index = 0
    while index < len(text):
        index, quote = _quote_step(text, index, quote)
    return quote == "'"


def _quote_step(text: str, index: int, quote: str) -> tuple[int, str]:
    """Advance one character of a quoted run, or open or close one."""
    character = text[index]
    if quote == '"' and character == "\\":
        return index + 2, quote
    if character in {"'", '"'} and quote in {"", character}:
        return index + 1, "" if quote else character
    return index + 1, quote


def executable_commands(text: str) -> list[re.Match[str]]:
    """Return the executable Kani commands in one recipe, in order."""
    return [
        match
        for line in logical_lines(text)
        if (match := KANI_COMMAND_RE.match(line)) is not None
    ]


def terminal_arguments(arguments: str) -> str:
    """Return the arguments the shell actually passes.

    A shell comment is not part of the command: anything written after one
    never reaches the checker. Every reading that compares or counts arguments
    goes through this, so a commented-out option cannot be counted.
    """
    return terminal_command(arguments)


def harness_names(arguments: str) -> list[str]:
    """Return the ``--harness`` values in one command's arguments.

    The scan is over the shell's own words, not over joined text, so a quoted
    value that happens to contain ``--harness`` cannot invent a selector.
    ``--harness NAME`` and ``--harness=NAME`` are both read.
    """
    return _option_values(shell_words(arguments), "-", "--harness")


def package_names(arguments: str) -> list[str]:
    """Return the package selectors in one command's arguments."""
    return _option_values(shell_words(arguments), "-p", "--package")


def _option_values(words: list[str], short: str, long: str) -> list[str]:
    """Return the values `short` or `long` select from a shell word list."""
    found: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word in {short, long} and index + 1 < len(words):
            found.append(words[index + 1])
            index += 2
            continue
        found.extend(
            word[len(f"{name}=") :]
            for name in (short, long)
            if word.startswith(f"{name}=")
        )
        index += 1
    return found


def selections(text: str) -> list[tuple[str, list[str]]]:
    """Return each Kani command's package and harness arguments, in order.

    Raises
    ------
    RecipeError
        When a command cannot be counted as a gate: it ignores its own exit
        status, swallows a failure, or names no single package.
    """
    return [
        _selection(match)
        for match in executable_commands(text)
    ]


def _selection(match: re.Match[str]) -> tuple[str, list[str]]:
    """Return one command's package and harness arguments, or raise."""
    arguments = terminal_command(match["arguments"])
    _reject_masking(match, arguments)
    packages = package_names(arguments)
    if len(packages) != 1:
        raise RecipeError(
            "each Kani recipe line must name exactly one package with -p, "
            f"got {packages!r} in {arguments.strip()!r}"
        )
    return packages[0], harness_names(arguments)


def _reject_masking(match: re.Match[str], arguments: str) -> None:
    """Reject a recipe line whose own failure could be swallowed."""
    if "-" in match["prefixes"]:
        raise RecipeError(
            f"a Kani recipe line is prefixed with '-' so its failure is "
            f"ignored: {arguments.strip()!r}"
        )
    if masked(arguments):
        raise RecipeError(
            f"a Kani recipe line swallows its failure: {arguments.strip()!r}"
        )
    operators = control_operators(arguments)
    if operators:
        raise RecipeError(
            "a Kani recipe line must be one command, but it contains the "
            f"shell control operator(s) {sorted(set(operators))!r}, "
            "through which a later command can hide the checker's status: "
            f"{arguments.strip()!r}"
        )


def wrappers(text: str) -> list[re.Match[str]]:
    """Return each logical line's checker wrapper, in order."""
    return [
        match
        for line in logical_lines(text)
        if (match := CHECKER_WRAPPER_RE.match(line)) is not None
    ]


#: The Kani options the `kani_suite` checker refuses, because each one hides,
#: redirects or rewrites the per-harness result blocks it parses.
FORBIDDEN_FLAGS = (
    "--quiet",
    "-q",
    "--output-into-files",
    "--bin",
    "--bins",
    "--no-codegen",
    "--only-codegen",
    "--log-file",
    "--tests",
    "-j",
    "--jobs",
)


def _spells(word: str, flag: str) -> bool:
    """Report whether `word` is `flag`, in any spelling the child accepts.

    `-j`, `-j8` and `-j=8` are one option. The checker refuses all three, so
    the contract must read them as one; matching the bare flag alone would let
    an attached short-option value pass as an ordinary argument.
    """
    if word == flag or word.startswith(f"{flag}="):
        return True
    if not _is_short_flag(flag):
        return False
    return word.startswith(flag) and len(word) > 2


def _is_short_flag(flag: str) -> bool:
    """Report whether `flag` is a single-dash short option such as ``-j``."""
    if len(flag) != 2:
        return False
    if not flag.startswith("-"):
        return False
    return not flag.startswith("--")


def shell_words(arguments: str) -> list[str]:
    """Return one command's words as the shell would pass them.

    Quotes group words and are removed, so ``--output-format "regular"``
    names the same value as the unquoted spelling. A command whose quoting is
    unbalanced falls back to a plain split, which can only over-report.
    """
    try:
        return shlex.split(terminal_arguments(arguments))
    except ValueError:
        return terminal_arguments(arguments).split()


def violations(arguments: str) -> list[str]:
    """Return the refused options in one Kani command's arguments, in order.

    `--output-format` is refused unless its value is ``regular``; every other
    refused option is refused whatever its value. The scan is token-wise over
    the command the shell runs, so a flag written after a shell comment is not
    read as one: the checker never receives it, and reporting it would fail a
    recipe for an option it does not pass.
    """
    words = shell_words(arguments)
    found = [
        word
        for word in words
        if any(_spells(word, flag) for flag in FORBIDDEN_FLAGS)
    ]
    for index, word in enumerate(words):
        if word == "--output-format":
            value = words[index + 1] if index + 1 < len(words) else ""
            if value != "regular":
                found.append(f"--output-format {value}".strip())
        elif word.startswith("--output-format=") and word != "--output-format=regular":
            found.append(word)
    return found
