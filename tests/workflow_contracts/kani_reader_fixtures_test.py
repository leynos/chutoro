"""Adversarial fixtures for the Kani source reader.

Every case here builds a tiny package whose Rust source spells a construct the
inventory reader has to classify, and asserts what :func:`discover` makes of
it: a proof inside a comment or a string literal, a proof behind a guard, a
proof inside a macro definition, a package whose only crate root is a binary,
and a proof-bearing file no module reaches. The path-attribute cases live in
:mod:`kani_path_fixtures_test`, beside the package writer both modules share in
:mod:`kani_reader_support`.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

from kani_harness_reading import Inventory, UnresolvedDeclaration, discover
from kani_manifest_test import _validate_entry
from kani_reader_support import _kinds, _names, _write_package

def test_reader_ignores_a_proof_in_a_comment(tmp_path: Path) -> None:
    """A proof written in a comment is a mention, not a declaration."""
    _write_package(
        tmp_path,
        "// #[kani::proof]\n// fn verify_commented() {}\n",
    )
    names = _names(discover(tmp_path))
    assert names == [], f"a comment must not declare a proof, got {names!r}"

def test_reader_ignores_a_proof_in_a_string_literal(tmp_path: Path) -> None:
    """A proof written inside a string literal is data, not a declaration."""
    _write_package(
        tmp_path,
        'const DOC: &str = "#[kani::proof]\\nfn verify_quoted() {}";\n',
    )
    names = _names(discover(tmp_path))
    assert names == [], f"a comment must not declare a proof, got {names!r}"

def test_reader_derives_a_nested_inline_module_path(tmp_path: Path) -> None:
    """A proof's harness path includes every inline module enclosing it."""
    _write_package(
        tmp_path,
        "mod outer {\n    mod inner {\n        #[kani::proof]\n"
        "        fn verify_nested() {}\n    }\n}\n",
    )
    names = _names(discover(tmp_path))
    assert names == ["outer::inner::verify_nested"], (
        f"the nested inline path must be derived, got {names!r}"
    )

def test_reader_derives_a_file_based_module_path(tmp_path: Path) -> None:
    """A file reached by ``mod name;`` contributes its module segment."""
    crate_root = _write_package(tmp_path, "mod proofs;\n")
    (crate_root.parent / "proofs.rs").write_text(
        "#[kani::proof]\nfn verify_file() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_file"], (
        f"the file-based module path must be derived, got {names!r}"
    )

def test_reader_reports_a_cfg_attr_wrapped_proof(tmp_path: Path) -> None:
    """A proof whose existence depends on configuration cannot be classified."""
    _write_package(
        tmp_path,
        '#[cfg_attr(kani, kani::proof)]\nfn verify_conditional() {}\n',
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["cfg-attr"], (
        f"a cfg_attr-wrapped proof must be reported, got {kinds!r}"
    )

def test_reader_reports_a_proof_inside_a_macro(tmp_path: Path) -> None:
    """A proof that only exists after expansion cannot be attributed."""
    _write_package(
        tmp_path,
        "macro_rules! declare {\n    () => {\n        #[kani::proof]\n"
        "        fn verify_expanded() {}\n    };\n}\n",
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["macro"], (
        f"a proof inside a macro must be reported, got {kinds!r}"
    )

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
    names = _names(inventory)
    assert names == [], f"an unreached proof must not be listed, got {names!r}"
    kinds = _kinds(inventory)
    assert kinds == ["unreachable"], (
        f"the orphaned source must be reported, got {kinds!r}"
    )

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

def test_reader_accepts_a_root_level_proof_identity(tmp_path: Path) -> None:
    """A proof at a crate's root has no module segment.

    Its harness path is the bare function name, so manifest validation must
    accept a name without ``::`` as well as one with it; requiring a module
    would reject the identity the reader reports and the comparison could
    never agree for such a package.
    """
    names = _names(
        discover(_write_package(tmp_path, "#[kani::proof]\nfn verify_root() {}\n").parent.parent)
    )
    assert names == ["verify_root"], (
        f"a root-level proof keeps its bare name, got {names!r}"
    )
    entries = [
        {"package": "fixture", "pretty_name": name, "tier": "fast"}
        for name in names
    ]
    for index, entry in enumerate(entries):
        _validate_entry(index, entry)

def test_reader_finds_a_proof_in_a_binary_target(tmp_path: Path) -> None:
    """A binary-only package's proofs compile, so they are read.

    Cargo discovers ``src/bin/*.rs`` without a manifest entry, and a package
    may declare its binary root explicitly. Neither root is a library, so a
    library-only sweep would drop the package and its proofs silently.
    """
    directory = tmp_path / "binary"
    (directory / "src" / "bin").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "binary"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (directory / "src" / "bin" / "prover.rs").write_text(
        "#[kani::proof]\nfn verify_binary() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["verify_binary"], (
        f"a discovered binary target must be read, got {names!r}"
    )

def test_reader_finds_a_proof_in_an_explicit_binary_target(tmp_path: Path) -> None:
    """An explicit ``[[bin]] path`` is a crate root like any other."""
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
    names = _names(discover(tmp_path))
    assert names == ["verify_declared"], (
        f"an explicit [[bin]] path must be read, got {names!r}"
    )

def test_reader_reads_a_proof_after_an_inline_module_opener(
    tmp_path: Path,
) -> None:
    """``mod proofs { #[kani::proof]`` puts the attribute on the opener's line.

    Rust gives no meaning to the line break before an item, so a reader that
    worked line by line would stop at ``mod`` and lose the proof.
    """
    _write_package(
        tmp_path,
        "mod proofs { #[kani::proof]\n    fn verify_inline() {}\n}\n",
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_inline"], (
        f"an attribute on the module opener's line must be read, got {names!r}"
    )

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
    names = _names(inventory)
    assert names == [], f"a helper module holds no proof, got {names!r}"
    kinds = _kinds(inventory)
    assert kinds == [], f"nothing must be reported, got {kinds!r}"

def test_reader_reads_two_attributes_sharing_a_line(tmp_path: Path) -> None:
    """``#[cfg(kani)] #[kani::proof]`` on one line still declares a proof.

    Rust gives no meaning to the line break between two outer attributes, so
    the two spellings are one declaration. Reading the line as a single
    attribute sees only ``cfg`` and loses the proof.
    """
    _write_package(
        tmp_path,
        "#[cfg(kani)] #[kani::proof]\nfn verify_same_line() {}\n",
    )
    names = _names(discover(tmp_path))
    assert names == ["verify_same_line"], (
        f"two attributes on one line must both be read, got {names!r}"
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

def test_reader_reports_a_proof_behind_a_test_guard(tmp_path: Path) -> None:
    """The gate compiles without ``cfg(test)``, so such a proof cannot run.

    Recording it as a harness would let the manifest require something the
    recipe can never verify.
    """
    _write_package(
        tmp_path,
        "#[cfg(all(kani, test))]\n#[kani::proof]\nfn verify_test_only() {}\n",
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"a test-gated proof must not be a harness, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == ["cfg-test"], (
        f"a test-gated proof must be reported, got {inventory.unresolved!r}"
    )

def test_reader_keeps_a_proof_behind_a_kani_guard(tmp_path: Path) -> None:
    """The negative control: ``#[cfg(kani)]`` is the gate's own configuration."""
    _write_package(
        tmp_path,
        "#[cfg(kani)]\n#[kani::proof]\nfn verify_gated() {}\n",
    )
    names = _names(discover(tmp_path))
    assert names == ["verify_gated"], f"a kani-gated proof must be read, got {names!r}"

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

def test_reader_reads_a_mod_declaration_split_over_lines(tmp_path: Path) -> None:
    """Rust permits whitespace, including a newline, inside ``mod name;``."""
    crate_root = _write_package(tmp_path, "mod\n    proofs\n;\n")
    (crate_root.parent / "proofs.rs").write_text(
        "#[kani::proof]\nfn verify_split() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_split"], (
        f"a split module declaration must be read, got {names!r}"
    )

def test_reader_reports_a_proof_in_a_parenthesised_macro(tmp_path: Path) -> None:
    """A macro body may be delimited by parentheses or brackets, not only braces."""
    _write_package(
        tmp_path,
        "macro_rules! declare (\n    () => {\n        #[kani::proof]\n"
        "        fn verify() {}\n    };\n);\n",
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["macro"], (
        f"a proof inside a parenthesised macro must be reported, got {kinds!r}"
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
