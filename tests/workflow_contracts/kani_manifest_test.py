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
import tomllib
import typing as typ
from pathlib import Path

import pytest
from kani_harness_reading import (
    Inventory,
    ProofDeclaration,
    UnresolvedDeclaration,
    discover,
)
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


class ManifestError(ValueError):
    """Raised when the manifest does not have the shape its consumers assume."""


def _document(path: Path = MANIFEST_PATH) -> dict[str, typ.Any]:
    """Return the parsed manifest document."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _entries(path: Path = MANIFEST_PATH) -> list[dict[str, typ.Any]]:
    """Return the manifest's harness entries, after validating their shape."""
    document = _document(path)
    unknown = set(document) - DOCUMENT_KEYS
    if unknown:
        raise ManifestError(f"unknown manifest keys: {sorted(unknown)}")
    entries = document.get("harness")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("harnesses.toml must declare at least one [[harness]]")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"harness {index} is not a table")
        _validate_entry(index, entry)
    return entries


def _validate_entry(index: int, entry: dict[str, typ.Any]) -> None:
    """Reject an entry whose keys, values or tier are not the expected shape."""
    missing = ENTRY_KEYS - set(entry)
    unknown = set(entry) - ENTRY_KEYS
    if missing:
        raise ManifestError(f"harness {index} is missing {sorted(missing)}")
    if unknown:
        raise ManifestError(f"harness {index} has unknown keys {sorted(unknown)}")
    for key in ("package", "pretty_name", "tier"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ManifestError(f"harness {index} has an empty {key}")
    if entry["tier"] not in TIERS:
        raise ManifestError(
            f"harness {index} has tier {entry['tier']!r}; expected one of "
            f"{sorted(TIERS)}"
        )
    if "::" not in entry["pretty_name"]:
        raise ManifestError(
            f"harness {index} names {entry['pretty_name']!r}, which is not a "
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
        f"reason it cannot be: {[(report.kind, report.path, report.line) for report in inventory.unresolved]!r}"
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
        if proof.path == "chutoro-core/src/distance/mod.rs"
    ]
    listed = [
        entry
        for entry in _entries()
        if entry["pretty_name"].startswith("distance::kani_proofs::")
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


# ---------------------------------------------------------------------------
# Adversarial fixtures: the reader's own contract
# ---------------------------------------------------------------------------

#: A package manifest, enough for the sweep to recognise a package.
PACKAGE_MANIFEST = '[package]\nname = "{name}"\nversion = "0.1.0"\n'


def _write_package(root: Path, source: str, name: str = "fixture") -> Path:
    """Create a one-file package under `root` and return its crate root."""
    directory = root / name
    (directory / "src").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        PACKAGE_MANIFEST.format(name=name), encoding="utf-8"
    )
    crate_root = directory / "src" / "lib.rs"
    crate_root.write_text(source, encoding="utf-8")
    return crate_root


def _names(inventory: Inventory) -> list[str]:
    """Return the harness names the inventory found, sorted."""
    return sorted(proof.pretty_name for proof in inventory.proofs)


def _kinds(inventory: Inventory) -> list[str]:
    """Return the kinds of unresolved report, sorted."""
    return sorted(report.kind for report in inventory.unresolved)


def test_reader_ignores_a_proof_in_a_comment(tmp_path: Path) -> None:
    """A proof written in a comment is a mention, not a declaration."""
    _write_package(
        tmp_path,
        "// #[kani::proof]\n// fn verify_commented() {}\n",
    )
    assert _names(discover(tmp_path)) == []


def test_reader_ignores_a_proof_in_a_string_literal(tmp_path: Path) -> None:
    """A proof written inside a string literal is data, not a declaration."""
    _write_package(
        tmp_path,
        'const DOC: &str = "#[kani::proof]\\nfn verify_quoted() {}";\n',
    )
    assert _names(discover(tmp_path)) == []


def test_reader_derives_a_nested_inline_module_path(tmp_path: Path) -> None:
    """A proof's harness path includes every inline module enclosing it."""
    _write_package(
        tmp_path,
        "mod outer {\n    mod inner {\n        #[kani::proof]\n"
        "        fn verify_nested() {}\n    }\n}\n",
    )
    assert _names(discover(tmp_path)) == ["outer::inner::verify_nested"]


def test_reader_derives_a_file_based_module_path(tmp_path: Path) -> None:
    """A file reached by ``mod name;`` contributes its module segment."""
    crate_root = _write_package(tmp_path, "mod proofs;\n")
    (crate_root.parent / "proofs.rs").write_text(
        "#[kani::proof]\nfn verify_file() {}\n", encoding="utf-8"
    )
    assert _names(discover(tmp_path)) == ["proofs::verify_file"]


def test_reader_reports_an_unresolvable_path_attribute(tmp_path: Path) -> None:
    """A ``#[path]`` naming no file is reported rather than skipped."""
    _write_package(tmp_path, '#[path = "absent.rs"]\nmod proofs;\n')
    assert _kinds(discover(tmp_path)) == ["path-attribute"]


def test_reader_follows_a_resolvable_path_attribute(tmp_path: Path) -> None:
    """A ``#[path]`` naming a real file still contributes its module's name.

    Rust takes the module's name from the declaration, not from the file's
    location, so a resolvable path attribute leaves the harness identity
    derivable. Reporting these instead would fail the contract on the
    repository's ordinary test-only ``#[path]`` modules.
    """
    crate_root = _write_package(tmp_path, '#[path = "elsewhere.rs"]\nmod proofs;\n')
    (crate_root.parent / "elsewhere.rs").write_text(
        "#[kani::proof]\nfn verify_moved() {}\n", encoding="utf-8"
    )
    assert _names(discover(tmp_path)) == ["proofs::verify_moved"]


def test_reader_reports_a_cfg_attr_wrapped_proof(tmp_path: Path) -> None:
    """A proof whose existence depends on configuration cannot be classified."""
    _write_package(
        tmp_path,
        '#[cfg_attr(kani, kani::proof)]\nfn verify_conditional() {}\n',
    )
    assert _kinds(discover(tmp_path)) == ["cfg-attr"]


def test_reader_reports_a_proof_inside_a_macro(tmp_path: Path) -> None:
    """A proof that only exists after expansion cannot be attributed."""
    _write_package(
        tmp_path,
        "macro_rules! declare {\n    () => {\n        #[kani::proof]\n"
        "        fn verify_expanded() {}\n    };\n}\n",
    )
    assert _kinds(discover(tmp_path)) == ["macro"]


def test_reader_reports_a_proof_bearing_source_no_module_reaches(
    tmp_path: Path,
) -> None:
    """An orphaned proof file is reported rather than silently dropped.

    Kani compiles a module only when the crate declares it, so a proof in an
    unreferenced file would never run. Reporting it keeps the manifest from
    promising a result the build cannot produce.
    """
    crate_root = _write_package(tmp_path, "")
    (crate_root.parent / "orphan.rs").write_text(
        "#[kani::proof]\nfn verify_orphan() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == []
    assert _kinds(inventory) == ["unreachable"]


def test_reader_keeps_a_module_file_without_a_proof_out_of_the_inventory(
    tmp_path: Path,
) -> None:
    """A reachable module holding no proof adds nothing to the inventory.

    The negative control for module traversal: it must not invent a
    declaration for a file it read but found nothing in.
    """
    crate_root = _write_package(tmp_path, "mod helper;\n")
    (crate_root.parent / "helper.rs").write_text(
        "pub fn work() -> bool { true }\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == []
    assert _kinds(inventory) == []


def test_unresolved_report_names_the_file_and_line(tmp_path: Path) -> None:
    """A report without a location would not be actionable in review."""
    _write_package(
        tmp_path,
        "// filler\nfn other() {}\n#[cfg_attr(kani, kani::proof)]\nfn verify_conditional() {}\n",
    )
    (report,) = discover(tmp_path).unresolved
    assert isinstance(report, UnresolvedDeclaration)
    assert report.line == 4, f"expected the declaration's line, got {report!r}"
    assert report.path.endswith("src/lib.rs"), (
        f"expected a source path, got {report!r}"
    )