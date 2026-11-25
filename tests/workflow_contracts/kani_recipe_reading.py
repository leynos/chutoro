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
from pathlib import Path

import pytest
from workflow_support import ROOT

MAKEFILE_PATH = ROOT / "Makefile"

#: The ``kani:`` target header and its immediately following tab-indented
#: recipe lines. The capture stops at the first line that is not tab-indented,
#: and deliberately does not stop at a blank line: make permits a blank line
#: between recipe lines, and a comment line inside a recipe is recipe text the
#: shell runs. Ending the capture at either would silently read a shorter tier.
KANI_TARGET_RE = re.compile(
    r"^kani:[^\n]*\n(?P<recipe>(?:(?:\t[^\n]*)?(?:\n|$))*)",
    re.MULTILINE,
)

#: The same shape for the ``kani-full:`` target.
KANI_FULL_TARGET_RE = re.compile(
    r"^kani-full:[^\n]*\n(?P<recipe>(?:(?:\t[^\n]*)?(?:\n|$))*)",
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
    r"(?:\$\([A-Z_]+\)\s+)*\$\(CARGO\)\s+kani\b(?P<arguments>.*)$"
)

#: The checker wrapper around one delegated command: the manifest, the
#: package and the scope, in that order, then the argument separator.
CHECKER_WRAPPER_RE = re.compile(
    r"^\t[-@+]*\$\(KANI_SUITE\)\s+--manifest\s+\$\(KANI_MANIFEST\)\s+"
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
#: for the shell.
SWALLOWING_SUFFIXES = ("|| true", "||true", "|| :", "||:")


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
    that ends in a backslash continues; the continuation's leading tab is kept
    and is matched by the patterns' whitespace.
    """
    logical: list[str] = []
    pending = ""
    for line in text.splitlines():
        continued = line.endswith("\\")
        pending += line[:-1] if continued else line
        if not continued:
            logical.append(pending)
            pending = ""
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


def names(arguments: str, pattern: re.Pattern[str]) -> list[str]:
    """Return the values `pattern` finds in one command's arguments."""
    return [found["name"] for found in pattern.finditer(arguments)]


def selections(text: str) -> list[tuple[str, list[str]]]:
    """Return each Kani command's package and harness arguments, in order.

    Raises
    ------
    RecipeError
        When a command cannot be counted as a gate: it ignores its own exit
        status, swallows a failure, or names no single package.
    """
    selected: list[tuple[str, list[str]]] = []
    for match in executable_commands(text):
        arguments = match["arguments"]
        if "-" in match["prefixes"]:
            raise RecipeError(
                f"a Kani recipe line is prefixed with '-' so its failure is "
                f"ignored: {arguments.strip()!r}"
            )
        if arguments.rstrip().endswith(SWALLOWING_SUFFIXES):
            raise RecipeError(
                f"a Kani recipe line swallows its failure: {arguments.strip()!r}"
            )
        packages = names(arguments, PACKAGE_ARGUMENT_RE)
        if len(packages) != 1:
            raise RecipeError(
                "each Kani recipe line must name exactly one package with -p, "
                f"got {packages!r} in {arguments.strip()!r}"
            )
        selected.append((packages[0], names(arguments, HARNESS_ARGUMENT_RE)))
    return selected


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
    "--no-codegen",
    "--only-codegen",
    "--log-file",
    "--tests",
    "-j",
    "--jobs",
)


def violations(arguments: str) -> list[str]:
    """Return the refused options in one Kani command's arguments, in order.

    `--output-format` is refused unless its value is ``regular``; every other
    refused option is refused whatever its value. The scan is token-wise, so a
    flag in a comment or a quoted argument has already been excluded by
    :func:`executable_commands`.
    """
    words = arguments.split()
    found = [
        word for word in words if word.split("=", 1)[0] in FORBIDDEN_FLAGS
    ]
    for index, word in enumerate(words):
        if word == "--output-format":
            value = words[index + 1] if index + 1 < len(words) else ""
            if value != "regular":
                found.append(f"--output-format {value}".strip())
        elif word.startswith("--output-format=") and word != "--output-format=regular":
            found.append(word)
    return found
