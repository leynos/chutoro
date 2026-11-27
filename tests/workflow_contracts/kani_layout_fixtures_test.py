"""Fixtures for proof discovery across Cargo package layouts.

A proof's harness identity depends on which crate Cargo builds and where its
root lives, and the reader has to agree with Cargo rather than with convention:
an explicit ``[lib] path``, a binary root outside ``src``, a name-only
``[[bin]]`` that resolves to ``src/main.rs`` or ``src/bin/<name>.rs``,
``autobins``/``autolib`` switching discovery off, and a build directory that
must not be swept. Each case here builds one such package and asserts what
:func:`discover` makes of it. The attribute, macro and guard cases live in
:mod:`kani_reader_fixtures_test`.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ
from pathlib import Path

from kani_harness_reading import discover
from kani_reader_support import _kinds, _names


def test_reader_finds_a_proof_in_a_package_with_an_explicit_lib_path(
    tmp_path: Path,
) -> None:
    """A package whose library lives outside ``src/`` is still read.

    Cargo honours ``[lib] path``, so the package compiles a proof that a
    ``src/lib.rs``-only sweep never looks for. Skipping it would omit the
    declaration without a report.
    """
    directory = tmp_path / "explicit"
    (directory / "proofs").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "explicit"\nversion = "0.1.0"\n\n'
        '[lib]\npath = "proofs/root.rs"\n',
        encoding="utf-8",
    )
    (directory / "proofs" / "root.rs").write_text(
        "#[kani::proof]\nfn verify_explicit() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    names = _names(inventory)
    assert names == ["verify_explicit"], (
        f"the explicitly pathed library root must be read, got {names!r}"
    )


def test_reader_finds_a_proof_in_a_binary_target(tmp_path: Path) -> None:
    """A binary-only package's proofs are reported, not inventoried.

    Cargo discovers ``src/bin/*.rs`` without a manifest entry, so a
    library-only sweep would miss the file entirely. The gate runs `cargo
    kani --lib`, though, so a proof there never executes: it is reported
    rather than certified as coverage.
    """
    directory = tmp_path / "binary"
    (directory / "src" / "bin").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "binary"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (directory / "src" / "bin" / "prover.rs").write_text(
        "#[kani::proof]\nfn verify_binary() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"a binary-target proof is not gate coverage, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == ["binary-target"], (
        f"a binary-target proof must be reported, got {inventory.unresolved!r}"
    )


def test_reader_finds_a_proof_in_an_explicit_binary_target(tmp_path: Path) -> None:
    """An explicit ``[[bin]] path`` is reported like a discovered one."""
    directory = tmp_path / "declared"
    (directory / "proofs").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "declared"\nversion = "0.1.0"\n\n'
        '[[bin]]\nname = "prover"\npath = "proofs/main.rs"\n',
        encoding="utf-8",
    )
    (directory / "proofs" / "main.rs").write_text(
        "#[kani::proof]\nfn verify_declared() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"an explicit binary target is not gate coverage, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == ["binary-target"], (
        f"an explicit binary target must be reported, got {inventory.unresolved!r}"
    )


def test_reader_reports_an_orphan_beside_a_custom_root(tmp_path: Path) -> None:
    """The orphan sweep covers custom source directories, not only ``src``.

    A library at ``[lib] path = "proofs/root.rs"`` keeps its sources outside
    ``src``; a proof-bearing file beside it that no module reaches is reported
    by the same sweep as one under ``src``.
    """
    directory = tmp_path / "custom"
    (directory / "proofs").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "custom"\nversion = "0.1.0"\n\n'
        '[lib]\npath = "proofs/root.rs"\n',
        encoding="utf-8",
    )
    (directory / "proofs" / "root.rs").write_text("", encoding="utf-8")
    (directory / "proofs" / "orphan.rs").write_text(
        "#[kani::proof]\nfn verify_orphan() {}\n", encoding="utf-8"
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["unreachable"], (
        f"an orphan beside a custom root must be reported, got {kinds!r}"
    )


def test_reader_finds_a_package_named_binary_beside_main(
    tmp_path: Path,
) -> None:
    """Cargo also accepts an existing ``src/bin/<package>.rs`` for that name.

    The inferred name accepts either conventional location, so the reader must
    take the one that exists rather than insisting on ``src/main.rs``.
    """
    directory = tmp_path / "beside"
    (directory / "src" / "bin").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "beside"\nversion = "0.1.0"\nautobins = false\n\n'
        '[[bin]]\nname = "beside"\n',
        encoding="utf-8",
    )
    (directory / "src" / "bin" / "beside.rs").write_text(
        "#[kani::proof]\nfn verify_beside() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _kinds(inventory) == ["binary-target"], (
        f"the existing named binary must be read, got {inventory.unresolved!r}"
    )


def test_reader_resolves_a_package_named_binary_to_main(
    tmp_path: Path,
) -> None:
    """A name-only ``[[bin]]`` matching the package reads ``src/main.rs``.

    Cargo's inference for that name is the package's own ``src/main.rs``, not
    ``src/bin/<name>.rs``; reading the wrong path would leave the package with
    no binary root and its proofs unread.
    """
    directory = tmp_path / "selfnamed"
    (directory / "src").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "selfnamed"\nversion = "0.1.0"\nautobins = false\n\n'
        '[[bin]]\nname = "selfnamed"\n',
        encoding="utf-8",
    )
    (directory / "src" / "main.rs").write_text(
        "#[kani::proof]\nfn verify_main() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _kinds(inventory) == ["binary-target"], (
        f"the package-named binary must be read, got {inventory.unresolved!r}"
    )


def test_reader_infers_a_named_binary_before_disabling_discovery(
    tmp_path: Path,
) -> None:
    """A declared ``[[bin]]`` resolves its path even when ``autobins`` is off.

    Cargo infers ``src/bin/<name>.rs`` for a name-only declaration because the
    caller asked for that target; ``autobins`` only turns off the targets
    nobody declared. Dropping it would leave a binary-only package with no
    root at all and its proofs unread.
    """
    directory = tmp_path / "named"
    (directory / "src" / "bin").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "named"\nversion = "0.1.0"\nautobins = false\n\n'
        '[[bin]]\nname = "tool"\n',
        encoding="utf-8",
    )
    (directory / "src" / "bin" / "tool.rs").write_text(
        "#[kani::proof]\nfn verify_tool() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _kinds(inventory) == ["binary-target"], (
        f"a named binary target must be read and reported, got {inventory.unresolved!r}"
    )


def test_reader_skips_an_implicit_library_when_autolib_is_false(
    tmp_path: Path,
) -> None:
    """A binary that loads ``src/lib.rs`` as a module has no library target.

    ``autolib = false`` tells Cargo not to build that file as a library, so
    reading it as a crate root would visit the same source twice.
    """
    directory = tmp_path / "noautolib"
    (directory / "src").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "noautolib"\nversion = "0.1.0"\nautolib = false\n\n'
        '[[bin]]\nname = "prover"\npath = "src/main.rs"\n',
        encoding="utf-8",
    )
    (directory / "src" / "main.rs").write_text(
        '#[path = "lib.rs"]\nmod proofs;\n', encoding="utf-8"
    )
    (directory / "src" / "lib.rs").write_text(
        "#[kani::proof]\nfn verify_loaded() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _kinds(inventory) == ["binary-target"], (
        f"only the binary target may be reported, got {inventory.unresolved!r}"
    )


def test_reader_prunes_a_build_directory_beside_a_custom_root(
    tmp_path: Path,
) -> None:
    """The orphan sweep never enters a build directory.

    A library at a custom path puts the package directory in the sweep, and a
    generated source under ``target`` there is not the repository's to read.
    """
    directory = tmp_path / "custombuild"
    (directory / "proofs" / "target" / "debug").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "custombuild"\nversion = "0.1.0"\n\n'
        '[lib]\npath = "proofs/root.rs"\n',
        encoding="utf-8",
    )
    (directory / "proofs" / "root.rs").write_text("", encoding="utf-8")
    (directory / "proofs" / "target" / "debug" / "generated.rs").write_text(
        "#[kani::proof]\nfn verify_generated() {}\n", encoding="utf-8"
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == [], (
        f"a generated source must not be an orphan, got {kinds!r}"
    )


def test_reader_skips_binaries_when_autobins_is_false(tmp_path: Path) -> None:
    """A package that disables discovery does not have those binaries.

    The file is loaded as a module on purpose, so reading it again as a binary
    would visit the same source twice and report a repeat that is not there.
    """
    directory = tmp_path / "noautobins"
    (directory / "src" / "bin").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "noautobins"\nversion = "0.1.0"\nautobins = false\n',
        encoding="utf-8",
    )
    (directory / "src" / "lib.rs").write_text(
        '#[path = "bin/helper.rs"]\nmod helper;\n', encoding="utf-8"
    )
    (directory / "src" / "bin" / "helper.rs").write_text(
        "#[kani::proof]\nfn verify_helper() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _kinds(inventory) == [], (
        f"a disabled binary target must not be read twice, got {inventory.unresolved!r}"
    )


def test_reader_sweeps_orphans_beside_every_crate_root(tmp_path: Path) -> None:
    """A binary root's directory is swept as well as ``src``.

    The package holds a library at ``src`` and a binary at ``proofs/main.rs``;
    an unreferenced proof beside the binary must still be reported.
    """
    directory = tmp_path / "twinned"
    (directory / "src").mkdir(parents=True)
    (directory / "proofs").mkdir()
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "twinned"\nversion = "0.1.0"\n\n'
        '[[bin]]\nname = "prover"\npath = "proofs/main.rs"\n',
        encoding="utf-8",
    )
    (directory / "src" / "lib.rs").write_text("", encoding="utf-8")
    (directory / "proofs" / "main.rs").write_text("", encoding="utf-8")
    (directory / "proofs" / "orphan.rs").write_text(
        "#[kani::proof]\nfn verify_orphan() {}\n", encoding="utf-8"
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["unreachable"], (
        f"an orphan beside a binary root must be reported, got {kinds!r}"
    )
