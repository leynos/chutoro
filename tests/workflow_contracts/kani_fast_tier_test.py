"""Contract between the Makefile's Kani tiers and the proof inventory.

``make kani`` is the pull-request gate, so its meaning depends on exactly
which harnesses the ``kani:`` target runs, and ``make kani-full`` is the
nightly tier, so its meaning depends on which packages it sweeps. Both are
compared against ``tools/kani/harnesses.toml``: the fast selectors must be
exactly the manifest's fast-tier harnesses for the package they name, and the
full tier must sweep each package that holds a proof.

The selectors are read from executable recipe lines only. A ``--harness``
written in a comment, echoed, or printed by ``printf`` is text, not a proof
selection, so the reading rejects all three. A recipe line that masks its own
failure — a trailing ``|| true`` or a leading ``-`` — is rejected too, because
a tier whose recipes cannot fail is not a gate.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections
import re
import typing as typ
from pathlib import Path

import pytest
from kani_manifest_test import _entries
from workflow_support import ROOT

MAKEFILE_PATH = ROOT / "Makefile"

#: The ``kani:`` target header and its immediately following tab-indented
#: recipe lines; the capture stops at the first line that is not tab-indented.
KANI_TARGET_RE = re.compile(
    r"^kani:[^\n]*\n(?P<recipe>(?:\t[^\n]*(?:\n|$))*)",
    re.MULTILINE,
)

#: The same shape for the ``kani-full:`` target.
KANI_FULL_TARGET_RE = re.compile(
    r"^kani-full:[^\n]*\n(?P<recipe>(?:\t[^\n]*(?:\n|$))*)",
    re.MULTILINE,
)

#: An executable Kani command in a tier recipe. The optional make-variable
#: prefixes cover the environment and the checker wrapper; a ``#`` comment, an
#: ``echo``, or a ``printf`` cannot satisfy the pattern. A leading ``-``, which
#: tells make to ignore the recipe's failure, is captured so the caller can
#: reject it.
KANI_COMMAND_RE = re.compile(
    r"^\t(?P<ignore>-)?(?:\$\([A-Z_]+\)\s+)*\$\(CARGO\)\s+kani\b(?P<arguments>.*)$"
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


def _recipe(text: str, pattern: re.Pattern[str], target: str) -> str:
    """Return the recipe body of one target, or raise."""
    match = pattern.search(text)
    if match is None or not match["recipe"]:
        raise RecipeError(f"the Makefile must define a {target}: target with a recipe")
    return match["recipe"].rstrip("\n")


def _executable_commands(recipe: str) -> list[re.Match[str]]:
    """Return the executable Kani commands in one recipe, in order."""
    return [
        match
        for line in recipe.splitlines()
        if (match := KANI_COMMAND_RE.match(line)) is not None
    ]


def _names(arguments: str, pattern: re.Pattern[str]) -> list[str]:
    """Return the values `pattern` finds in one command's arguments."""
    return [found["name"] for found in pattern.finditer(arguments)]


def _selections(recipe: str) -> list[tuple[str, list[str]]]:
    """Return each Kani command's package and harness arguments, in order.

    Raises
    ------
    RecipeError
        When a command cannot be counted as a gate: it ignores its own exit
        status, swallows a failure, or names no single package.
    """
    selected: list[tuple[str, list[str]]] = []
    for match in _executable_commands(recipe):
        arguments = match["arguments"]
        if match["ignore"] is not None:
            raise RecipeError(
                f"a Kani recipe line is prefixed with '-' so its failure is "
                f"ignored: {arguments.strip()!r}"
            )
        if arguments.rstrip().endswith(SWALLOWING_SUFFIXES):
            raise RecipeError(
                f"a Kani recipe line swallows its failure: {arguments.strip()!r}"
            )
        packages = _names(arguments, PACKAGE_ARGUMENT_RE)
        if len(packages) != 1:
            raise RecipeError(
                "each Kani recipe line must name exactly one package with -p, "
                f"got {packages!r} in {arguments.strip()!r}"
            )
        selected.append((packages[0], _names(arguments, HARNESS_ARGUMENT_RE)))
    return selected


def _manifest_fast(package: str) -> list[str]:
    """Return the manifest's fast-tier harnesses for `package`, sorted."""
    return sorted(
        entry["pretty_name"]
        for entry in _entries()
        if entry["package"] == package and entry["tier"] == "fast"
    )


def _manifest_packages() -> list[str]:
    """Return every package the manifest names, sorted."""
    return sorted({entry["package"] for entry in _entries()})


@pytest.fixture(scope="module")
def makefile() -> str:
    """Return the Makefile's text."""
    return MAKEFILE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fast_recipe(makefile: str) -> str:
    """Return the recipe body of the ``kani:`` target."""
    return _recipe(makefile, KANI_TARGET_RE, "kani")


@pytest.fixture(scope="module")
def full_recipe(makefile: str) -> str:
    """Return the recipe body of the ``kani-full:`` target."""
    return _recipe(makefile, KANI_FULL_TARGET_RE, "kani-full")


@pytest.mark.parametrize(
    "recipe",
    (
        "\t# $(CARGO) kani --harness verify_mst_structural_correctness_4_nodes",
        "\techo $(CARGO) kani --harness verify_mst_structural_correctness_4_nodes",
        "\tprintf '%s\\n' '--harness verify_mst_structural_correctness_4_nodes'",
        '\t@echo "$(CARGO) kani -p chutoro-core --harness verify_x"',
    ),
)
def test_nonexecuting_recipe_text_selects_no_harness(recipe: str) -> None:
    """Names in comments or in output commands cannot satisfy the contract."""
    assert _executable_commands(recipe) == []


@pytest.mark.parametrize(
    ("recipe", "forbidden"),
    (
        ("\t$(CARGO) kani -p chutoro-core || true", "|| true"),
        ("\t$(CARGO) kani -p chutoro-core || :", "|| :"),
        ("\t-$(CARGO) kani -p chutoro-core", "prefixed with '-'"),
        ("\t$(CARGO) kani -p chutoro-core --harness verify_x ||true", "||true"),
    ),
)
def test_masked_recipe_lines_are_rejected(recipe: str, forbidden: str) -> None:
    """A tier whose recipe cannot fail is not a gate."""
    with pytest.raises(RecipeError) as raised:
        _selections(recipe)
    assert forbidden in str(raised.value), (
        f"the rejection must name the masking ({forbidden!r}), got {raised.value}"
    )


def test_fast_recipe_runs_only_executable_kani_commands(fast_recipe: str) -> None:
    """Every fast-tier line is an executable Kani command with one package."""
    selections = _selections(fast_recipe)
    assert selections, (
        "the kani: target must run at least one executable Kani command"
    )
    assert all(name for _, names in selections for name in names), (
        "every fast-tier command must name the harness it runs; an unselected "
        f"command runs a whole package in the pull-request tier: {selections!r}"
    )


def test_fast_selectors_are_exactly_the_manifest_fast_tier(
    fast_recipe: str,
) -> None:
    """Each package's fast selectors are exactly that package's fast proofs.

    Compared as multisets, so a dropped selector and a duplicated one are both
    failures. The selectors are fully qualified: a bare function name still
    matches under ``--harness``, but it would also match a future sibling in
    another module, which is what the manifest exists to prevent.
    """
    selected = collections.Counter(
        name for _, names in _selections(fast_recipe) for name in names
    )
    expected = collections.Counter(
        name for package in _manifest_packages() for name in _manifest_fast(package)
    )
    missing = sorted((expected - selected).elements())
    unexpected = sorted((selected - expected).elements())
    duplicated = sorted(
        name for name, count in selected.items() if count > 1
    )
    assert not missing, (
        "the kani: target must run every fast-tier harness the manifest "
        f"lists; missing: {missing!r}"
    )
    assert not unexpected, (
        "the kani: target runs harnesses the manifest does not classify as "
        f"fast-tier; unexpected: {unexpected!r}"
    )
    assert not duplicated, (
        "the kani: target runs these harnesses more than once, so one proof "
        f"result is being asked for twice: {duplicated!r}"
    )


def test_fast_selectors_are_qualified_paths(fast_recipe: str) -> None:
    """A fast selector must be the qualified path Kani prints.

    An unqualified selector is ambiguous under any future sibling, and it does
    not match the identity the manifest and the checker compare.
    """
    unqualified = sorted(
        name
        for _, names in _selections(fast_recipe)
        for name in names
        if "::" not in name
    )
    assert not unqualified, (
        "every fast-tier --harness selector must be the fully qualified path "
        f"the manifest lists; these are not: {unqualified!r}"
    )


def test_fast_selectors_are_grouped_by_their_own_package(fast_recipe: str) -> None:
    """A selector must be passed to the package that declares it.

    ``--harness`` is applied within ``-p``'s package, so a selector placed on
    another package's command matches nothing and the proof silently stops
    running.
    """
    misplaced: list[tuple[str, str]] = []
    for package, names in _selections(fast_recipe):
        allowed = set(_manifest_fast(package))
        misplaced.extend(
            (package, name) for name in names if name not in allowed
        )
    assert not misplaced, (
        "these selectors are passed to a package whose manifest fast tier does "
        f"not list them: {misplaced!r}"
    )


def test_fast_tier_covers_each_package_that_has_fast_proofs(
    fast_recipe: str,
) -> None:
    """A package with fast proofs must appear in the pull-request tier."""
    packages = {package for package, _ in _selections(fast_recipe)}
    populated = {
        package for package in _manifest_packages() if _manifest_fast(package)
    }
    assert packages == populated, (
        "the kani: target must run each package that has fast-tier proofs: "
        f"missing {sorted(populated - packages)}, extra {sorted(packages - populated)}"
    )


def test_full_tier_sweeps_every_package_holding_a_proof(full_recipe: str) -> None:
    """The nightly tier verifies each package's whole proof surface.

    ``make kani-full`` selects no harness, so it sweeps the package. Every
    package in the manifest must be swept, or a proof classified ``full``
    would never run anywhere.
    """
    swept = {package for package, _ in _selections(full_recipe)}
    expected = set(_manifest_packages())
    assert swept == expected, (
        "the kani-full: target must sweep exactly the packages that hold "
        f"proofs: missing {sorted(expected - swept)}, extra "
        f"{sorted(swept - expected)}"
    )


def test_full_tier_selects_no_individual_harness(full_recipe: str) -> None:
    """The nightly tier sweeps packages; per-harness selection belongs in fast.

    A harness selector here would make the tier depend on a list that the fast
    tier already owns, and a proof added to the manifest would then be
    silently excluded from the tier meant to cover everything.
    """
    selected = [
        (package, name)
        for package, names in _selections(full_recipe)
        for name in names
    ]
    assert not selected, (
        "the kani-full: target must not select individual harnesses; it "
        f"verifies the package's whole proof surface: {selected!r}"
    )