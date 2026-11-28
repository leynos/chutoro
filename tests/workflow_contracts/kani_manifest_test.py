"""Contract binding the Kani proof inventory to the declarations in the tree.

``tools/kani/harnesses.toml`` is the single classification of the proof
surface. This module checks two things about it, in both directions.

First, that the manifest is well formed: every entry names a package, a
qualified harness path, and one of the two tiers, and no identity appears
twice. A malformed entry is rejected here rather than being read differently
by each consumer.

Second, that the manifest and the Rust sources describe the same proofs. The
source scan comes from :mod:`kani_harness_reading`, which reads declarations
out of the tree, so an inline ``#[cfg(kani)] mod kani_proofs`` block counts
exactly as a ``kani_*.rs`` file does. A declaration with no manifest entry, a
manifest entry with no declaration, and a duplicated identity are reported
separately, because the three need different fixes.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections
import dataclasses
import re
import tomllib
import typing as typ
from pathlib import Path

import pytest
from kani_harness_reading import Inventory, ProofDeclaration, discover
from workflow_support import ROOT

MANIFEST_PATH = ROOT / "tools" / "kani" / "harnesses.toml"

#: The tiers a declaration may be classified into. ``fast`` runs on every pull
#: request; ``full`` runs only in the nightly package sweep.
TIERS = frozenset({"fast", "full"})

#: The only keys an entry may carry. An unknown key is rejected rather than
#: ignored, so a typo in a field name fails instead of silently dropping the
#: classification it was meant to set.
ENTRY_KEYS = frozenset({"package", "pretty_name", "tier"})

#: The only keys the document may carry.
DOCUMENT_KEYS = frozenset({"version", "harness"})

#: The only manifest version the Rust loader reads. A document that names any
#: other is rejected there, so accepting it here would report a healthy
#: manifest that the checker cannot load.
MANIFEST_VERSION = 1


class ManifestError(ValueError):
    """Raised when the manifest does not have the shape its consumers assume."""


def _document(path: Path = MANIFEST_PATH) -> dict[str, typ.Any]:
    """Return the parsed manifest document."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _entries(path: Path = MANIFEST_PATH) -> list[dict[str, typ.Any]]:
    """Return the manifest's harness entries, after validating their shape."""
    entries = _harness_list(_checked_document(path))
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"harness {index} is not a table")
        _validate_entry(index, entry)
    return entries


def _checked_document(path: Path) -> dict[str, typ.Any]:
    """Return the parsed document, rejecting keys and versions no consumer expects."""
    document = _document(path)
    unknown = set(document) - DOCUMENT_KEYS
    if unknown:
        raise ManifestError(f"unknown manifest keys: {sorted(unknown)}")
    version = document.get("version")
    # `True == 1` and `1.0 == 1` in Python, while the Rust loader reads a u32;
    # the type is checked before the value so those forms are rejected here.
    if not isinstance(version, int) or isinstance(version, bool):
        raise ManifestError(f"manifest version {version!r} is not an integer")
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"manifest version {version!r}; expected {MANIFEST_VERSION}"
        )
    return document


def _harness_list(document: dict[str, typ.Any]) -> list[typ.Any]:
    """Return the document's non-empty list of harness tables."""
    entries = document.get("harness")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("harnesses.toml must declare at least one [[harness]]")
    return entries


def _validate_entry(index: int, entry: dict[str, typ.Any]) -> None:
    """Reject an entry whose keys, values or tier are not the expected shape."""
    _check_keys(index, entry)
    _check_values(index, entry)


def _check_keys(index: int, entry: dict[str, typ.Any]) -> None:
    """Reject an entry with a missing or unknown key."""
    missing = ENTRY_KEYS - set(entry)
    unknown = set(entry) - ENTRY_KEYS
    if missing:
        raise ManifestError(f"harness {index} is missing {sorted(missing)}")
    if unknown:
        raise ManifestError(f"harness {index} has unknown keys {sorted(unknown)}")


def _check_values(index: int, entry: dict[str, typ.Any]) -> None:
    """Reject an empty field, an unknown tier or an unqualified name."""
    for key in ("package", "pretty_name", "tier"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ManifestError(f"harness {index} has an empty {key}")
    _check_tier(index, entry["tier"])
    _check_qualified(index, entry["pretty_name"])


def _check_tier(index: int, tier: typ.Any) -> None:
    """Reject a tier outside the two the classifier defines."""
    if tier not in TIERS:
        raise ManifestError(
            f"harness {index} has tier {tier!r}; expected one of "
            f"{sorted(TIERS)}"
        )


#: One Rust identifier segment of a harness path.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _check_qualified(index: int, pretty_name: typ.Any) -> None:
    """Reject a name that is not a qualified harness path.

    A proof declared at a crate's root has no module segment, so its harness
    path is the bare function name and is accepted: requiring ``::`` would
    reject the identity the reader reports for it and the manifest/source
    comparison could never agree for such a package.
    """
    if not all(_IDENTIFIER.fullmatch(segment) for segment in pretty_name.split("::")):
        raise ManifestError(
            f"harness {index} names {pretty_name!r}, which is not a "
            "qualified harness path"
        )


def _identity(entry: typ.Mapping[str, str]) -> tuple[str, str]:
    """Return the package-and-path identity of one entry or declaration."""
    return entry["package"], entry["pretty_name"]


def _declarations(inventory: Inventory) -> list[tuple[str, str]]:
    """Return each declared proof's identity, sorted."""
    return [_identity(entry) for entry in map(_as_mapping, inventory.proofs)]


def _as_mapping(proof: ProofDeclaration) -> dict[str, str]:
    """Return a declaration as the mapping shape the comparisons share."""
    return {"package": proof.package, "pretty_name": proof.pretty_name}


def _differences(
    declared: typ.Iterable[tuple[str, str]],
    listed: typ.Iterable[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return the items only in `declared` and only in `listed`, each sorted."""
    left, right = collections.Counter(declared), collections.Counter(listed)
    only_left = sorted((left - right).elements())
    only_right = sorted((right - left).elements())
    return only_left, only_right


def _duplicates(items: typ.Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return the identities appearing more than once, sorted."""
    counts = collections.Counter(items)
    return sorted(identity for identity, count in counts.items() if count > 1)


@pytest.fixture(scope="module")
def manifest() -> list[dict[str, typ.Any]]:
    """Return the validated manifest entries."""
    return _entries()


@pytest.fixture(scope="module")
def inventory() -> Inventory:
    """Return the proof surface read from the tree once for the whole module."""
    return discover()


def test_manifest_version_must_be_the_one_the_loader_reads(
    tmp_path: Path,
) -> None:
    """The Python contract must reject what the Rust loader rejects.

    A version the checker cannot load would otherwise pass every Python check
    and fail only when `make kani` runs.
    """
    for version in ("2", "true", "1.0"):
        path = tmp_path / f"harnesses-{version}.toml"
        path.write_text(
            f'version = {version}\n[[harness]]\npackage = "p"\n'
            'pretty_name = "a::b"\ntier = "fast"\n',
            encoding="utf-8",
        )
        with pytest.raises(ManifestError):
            _entries(path)


def test_manifest_entries_are_well_formed(manifest: list[dict[str, typ.Any]]) -> None:
    """Every entry names a package, a qualified proof, and a known tier."""
    assert manifest, "the manifest must classify at least one harness"
    tiers = {entry["tier"] for entry in manifest}
    assert tiers <= TIERS, f"unexpected tiers: {sorted(tiers - TIERS)}"


def test_manifest_lists_each_identity_once(
    manifest: list[dict[str, typ.Any]],
) -> None:
    """A duplicated identity would make the expected result set ambiguous."""
    duplicates = _duplicates(map(_identity, manifest))
    assert not duplicates, (
        "the manifest lists these identities more than once, so the checker "
        f"could not say how many results to expect: {duplicates!r}"
    )


def test_manifest_and_sources_describe_the_same_proofs(
    manifest: list[dict[str, typ.Any]], inventory: Inventory
) -> None:
    """Every declaration is classified, and every entry is a real declaration.

    The two directions are reported separately. A declaration missing from the
    manifest would never be verified by a checked run; an entry with no
    declaration would make a checked run wait for a harness that cannot
    appear.
    """
    declared = _declarations(inventory)
    listed = [_identity(entry) for entry in manifest]
    unlisted, undeclared = _differences(declared, listed)
    assert not unlisted, (
        "these proof declarations have no manifest entry, so a checked Kani "
        f"run would not expect them: {unlisted!r}"
    )
    assert not undeclared, (
        "these manifest entries have no proof declaration in the tree, so a "
        f"checked Kani run would wait for them: {undeclared!r}"
    )


def test_manifest_packages_match_the_packages_holding_proofs(
    manifest: list[dict[str, typ.Any]], inventory: Inventory
) -> None:
    """The package set is compared as a set, not merely entry by entry."""
    listed = {entry["package"] for entry in manifest}
    declared = {proof.package for proof in inventory.proofs}
    assert listed == declared, (
        "the manifest must name exactly the packages that hold proofs: "
        f"missing {sorted(declared - listed)}, extra {sorted(listed - declared)}"
    )


def test_source_scan_reports_no_unresolved_declaration(inventory: Inventory) -> None:
    """A construct the reader cannot attribute must fail the contract.

    An unresolved report is not a warning. Each one names a proof-shaped
    construct whose harness identity is unknown, so accepting it would leave
    part of the surface unclassified.
    """
    assert not inventory.unresolved, (
        "the proof scan could not attribute these constructs; each must be "
        "restructured so its harness path is derivable, or recorded with the "
        f"reason it cannot be: {[dataclasses.astuple(report) for report in inventory.unresolved]!r}"
    )


def test_scan_finds_the_inline_distance_proofs(inventory: Inventory) -> None:
    """Guard the derivation against regressing to a filename-only inventory.

    The distance proofs sit in an inline module inside ``distance/mod.rs``, a
    filename no ``kani_*.rs`` pattern would match. A scan that only looked at
    filenames would miss every one of them, and the manifest comparison would
    still pass because both sides would be equally blind — so the count is
    taken from the manifest's own distance entries rather than restated here.
    """
    distance = [
        proof
        for proof in inventory.proofs
        if proof.package == "chutoro-core"
        and proof.path == "chutoro-core/src/distance/mod.rs"
    ]
    listed = [
        entry
        for entry in _entries()
        if entry["package"] == "chutoro-core"
        and entry["pretty_name"].startswith("distance::kani_proofs::")
    ]
    assert listed, "the manifest must classify the distance proofs"
    assert len(distance) == len(listed), (
        "the scan must find every proof the manifest lists in the inline "
        f"distance::kani_proofs module: found {len(distance)}, listed "
        f"{len(listed)}"
    )
    assert {proof.package for proof in distance} == {"chutoro-core"}, (
        f"the distance proofs must be attributed to chutoro-core, got {distance!r}"
    )


def test_fast_tier_is_a_strict_subset(inventory: Inventory, manifest: list) -> None:
    """The fast tier must be non-empty and smaller than the whole surface."""
    fast = [entry["pretty_name"] for entry in manifest if entry["tier"] == "fast"]
    assert fast, "the manifest must classify some proofs as fast-tier"
    assert len(fast) < len(inventory.proofs), (
        "the fast tier must not be the whole proof surface; the nightly sweep "
        "exists to run the proofs left out of it"
    )
