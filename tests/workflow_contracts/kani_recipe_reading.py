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
from workflow_support import ROOT

MAKEFILE_PATH = ROOT / "Makefile"

#: The ``kani:`` target header and its immediately following recipe lines. The
#: capture stops at the first line that belongs to neither the recipe nor a
#: comment, and deliberately does not stop at a blank line or a column-0 ``#``
#: line: make permits a blank line between recipe lines, a tab-indented comment
#: is recipe text the shell runs, and a column-0 comment is a make comment that
#: does not end the recipe. Ending the capture at any of them would silently
#: read a shorter tier, and a failure-ignoring command after the comment would
#: go unchecked.
KANI_TARGET_RE = re.compile(
    r"^kani:[^\n]*\n(?P<recipe>(?:(?:\t[^\n]*|#[^\n]*)?(?:\n|$))*)",
    re.MULTILINE,
)

#: The same shape for the ``kani-full:`` target.
KANI_FULL_TARGET_RE = re.compile(
    r"^kani-full:[^\n]*\n(?P<recipe>(?:(?:\t[^\n]*|#[^\n]*)?(?:\n|$))*)",
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

#: A harness selector and its value, as ``--harness NAME`` or
#: ``--harness=NAME``.
HARNESS_ARGUMENT_RE = re.compile(
    r"(?:^|\s)--harness(?:=|\s+)(?P<name>[A-Za-z_][A-Za-z0-9_:]*)(?=\s|$)"
)

#: The package selector and its value.
PACKAGE_ARGUMENT_RE = re.compile(
    r"(?:^|\s)(?:-p|--package)(?:=|\s+)(?P<name>[A-Za-z0-9_-]+)(?=\s|$)"
)

#: Recipe text that swallows a failure. GNU make reads a leading ``-`` as
#: ignore-the-status, and a trailing ``|| true`` (or ``|| :``) does the same
#: for the shell. The shell ignores spacing around its tokens, so `||   true`
#: is the same operator; matching a fixed list of spellings would accept it.
_MASKING_SUFFIX = re.compile(r"\|\|\s*(?:true|:)\s*$")

#: Shell control operators this reader refuses. A tier recipe is one command
#: that make can fail on; a `;` lets a later `true` hide the checker's status,
#: and `&&`/`&`/`|` do the same in other shapes. Quoted text is exempt.
_CONTROL_OPERATORS = frozenset({";", "&&", "&", "|"})

#: A shell comment, which ends the command before anything written after it.
_SHELL_COMMENT = re.compile(r"(?:^|\s)#.*$")


def _terminal_command(arguments: str) -> str:
    """Return the command text with any trailing shell comment removed.

    A shell comment is not part of the command the shell runs, so nothing
    written after one is a package, a harness selector or an option. The scan
    therefore reads this text rather than the recipe's raw arguments: a
    commented-out `--harness` or `--jobs=2` would otherwise be counted as a
    selection or refused as a flag the command passes when it passes neither.

    Quote-aware enough for a recipe line, which carries no comment characters
    inside its quoted arguments in this repository.
    """
    without_comment = _SHELL_COMMENT.sub("", arguments)
    return without_comment.rstrip()


def _masked(command: str) -> bool:
    """Report whether `command` swallows its own failure."""
    return _MASKING_SUFFIX.search(command) is not None


def _control_operators(command: str) -> list[str]:
    """Return the unquoted shell control operators in `command`."""
    found: list[str] = []
    quote = ""
    index = 0
    while index < len(command):
        character = command[index]
        if quote:
            if character == quote:
                quote = ""
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
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


class RecipeError(ValueError):
    """Raised when a tier recipe cannot be read as a set of Kani selections."""


def recipe(text: str, pattern: re.Pattern[str], target: str) -> str:
    """Return the recipe body of one target, or raise."""
    match = pattern.search(text)
    if match is None or not match["recipe"]:
        raise RecipeError(f"the Makefile must define a {target}: target with a recipe")
    return match["recipe"].rstrip("\n")


def logical_lines(text: str) -> list[str]:
    """Return one recipe's commands with backslash continuations joined.

    A recipe splits a command across lines for readability; make joins them
    before the shell sees them, so the contract must join them too. A line
    that ends in a backslash continues, and make removes the recipe-prefix tab
    that begins the continuation: a `--harness=` whose value is on the
    next line reaches the shell as `--harness=verify`. One tab is removed, so
    any further indentation survives.
    """
    logical: list[str] = []
    pending = ""
    joining = False
    for line in text.splitlines():
        current = line[1:] if joining and line.startswith("\t") else line
        continued = current.endswith("\\")
        pending += current[:-1] if continued else current
        if not continued:
            logical.append(pending)
            pending = ""
        joining = continued
    if pending:
        logical.append(pending)
    return logical


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
    return _terminal_command(arguments)


def names(arguments: str, pattern: re.Pattern[str]) -> list[str]:
    """Return the values `pattern` finds in one command's arguments."""
    return [
        found["name"] for found in pattern.finditer(terminal_arguments(arguments))
    ]


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
    arguments = _terminal_command(match["arguments"])
    _reject_masking(match, arguments)
    packages = names(arguments, PACKAGE_ARGUMENT_RE)
    if len(packages) != 1:
        raise RecipeError(
            "each Kani recipe line must name exactly one package with -p, "
            f"got {packages!r} in {arguments.strip()!r}"
        )
    return packages[0], names(arguments, HARNESS_ARGUMENT_RE)


def _reject_masking(match: re.Match[str], arguments: str) -> None:
    """Reject a recipe line whose own failure could be swallowed."""
    if "-" in match["prefixes"]:
        raise RecipeError(
            f"a Kani recipe line is prefixed with '-' so its failure is "
            f"ignored: {arguments.strip()!r}"
        )
    if _masked(arguments):
        raise RecipeError(
            f"a Kani recipe line swallows its failure: {arguments.strip()!r}"
        )
    operators = _control_operators(arguments)
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
