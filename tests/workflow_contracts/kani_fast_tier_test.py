"""Contract between the Makefile's Kani tiers and the proof inventory.

``make kani`` is the pull-request gate, so its meaning depends on exactly
which harnesses the ``kani:`` target runs, and ``make kani-full`` is the
nightly tier, so its meaning depends on which packages it sweeps. Both are
compared against ``tools/kani/harnesses.toml``: the fast selectors must be
exactly the manifest's fast-tier harnesses for the package they name, and the
full tier must sweep each package that holds a proof.

The recipe reading, and the fixtures both tier contracts share, live in
:mod:`kani_recipe_reading`. The commands themselves are executed by
:mod:`kani_tier_routing_test`, which runs each tier against a recording
checker; this module asserts what a command must say, not that it runs.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections

import pytest

from kani_manifest_test import _entries
from kani_recipe_reading import (
    RecipeError,
    executable_commands,
    selections,
    violations,
)


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
    assert executable_commands(recipe) == []


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
        selections(recipe)
    assert forbidden in str(raised.value), (
        f"the rejection must name the masking ({forbidden!r}), got {raised.value}"
    )


def test_fast_recipe_runs_only_executable_kani_commands(fast_recipe: str) -> None:
    """Every fast-tier line is an executable Kani command with one package."""
    selected = selections(fast_recipe)
    assert selected, (
        "the kani: target must run at least one executable Kani command"
    )
    assert all(name for _, names in selected for name in names), (
        "every fast-tier command must name the harness it runs; an unselected "
        f"command runs a whole package in the pull-request tier: {selected!r}"
    )


@pytest.mark.parametrize(
    "arguments",
    (
        "-p chutoro-core --lib --quiet",
        "-p chutoro-core --lib -q",
        "-p chutoro-core --lib --output-format terse",
        "-p chutoro-core --lib --output-format=old",
        "-p chutoro-core --lib --output-into-files",
        "-p chutoro-core --lib --no-codegen",
        "-p chutoro-core --lib --only-codegen",
        "-p chutoro-core --lib --log-file kani.log",
        "-p chutoro-core --lib --tests",
        "-p chutoro-core --lib -j",
        "-p chutoro-core --lib --jobs 4",
    ),
)
def test_forbidden_kani_flags_are_rejected(arguments: str) -> None:
    """The checker refuses options that hide, redirect or rewrite its input."""
    assert violations(arguments), (
        f"expected {arguments!r} to contain a refused option"
    )


def test_a_regular_output_format_is_not_a_violation() -> None:
    """The negative control: the one output format the checker reads passes."""
    assert violations("-p chutoro-core --lib --output-format regular") == []


def test_fast_commands_pass_no_forbidden_flag(fast_recipe: str) -> None:
    """Every executable fast-tier command is one the checker will accept."""
    offenders = [
        (arguments, violations(arguments))
        for match in executable_commands(fast_recipe)
        if (arguments := match["arguments"]) and violations(arguments)
    ]
    assert not offenders, (
        "the kani: target passes options the checker refuses, so the run "
        f"would fail before verification: {offenders!r}"
    )


def test_full_commands_pass_no_forbidden_flag(full_recipe: str) -> None:
    """Every executable nightly command is one the checker will accept."""
    offenders = [
        (arguments, violations(arguments))
        for match in executable_commands(full_recipe)
        if (arguments := match["arguments"]) and violations(arguments)
    ]
    assert not offenders, (
        "the kani-full: target passes options the checker refuses, so the "
        f"run would fail before verification: {offenders!r}"
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
        name for _, names in selections(fast_recipe) for name in names
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
        for _, names in selections(fast_recipe)
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
    for package, names in selections(fast_recipe):
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
    packages = {package for package, _ in selections(fast_recipe)}
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
    swept = {package for package, _ in selections(full_recipe)}
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
        for package, names in selections(full_recipe)
        for name in names
    ]
    assert not selected, (
        "the kani-full: target must not select individual harnesses; it "
        f"verifies the package's whole proof surface: {selected!r}"
    )
