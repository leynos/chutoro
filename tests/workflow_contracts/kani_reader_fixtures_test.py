"""Adversarial fixtures for the Kani source reader.

Every case here builds a tiny package whose Rust source spells a construct the
inventory reader has to classify, and asserts what :func:`discover` makes of
it. The shapes are the ones a real tree can contain: a proof inside a comment
or a string literal, a ``#[path]`` that resolves, one that does not, one in a
raw string, one in a shape the reader cannot read, an inline module, a module
loaded from elsewhere, a proof reached only through a shared parent, and a
package whose only crate root is a binary.

They live apart from :mod:`kani_manifest_test` so each module stays inside the
file-length limit; that module keeps the repository-wide comparison between
``tools/kani/harnesses.toml`` and the tree.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

from kani_harness_reading import Inventory, UnresolvedDeclaration, discover
from kani_manifest_test import _validate_entry

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


def test_reader_reports_an_unresolvable_path_attribute(tmp_path: Path) -> None:
    """A ``#[path]`` naming no file is reported rather than skipped."""
    _write_package(tmp_path, '#[path = "absent.rs"]\nmod proofs;\n')
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["path-attribute"], (
        f"an unresolvable #[path] must be reported, got {kinds!r}"
    )


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
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_moved"], (
        f"a resolvable #[path] must contribute its module name, got {names!r}"
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


def test_reader_resolves_a_path_loaded_module_children_beside_it(
    tmp_path: Path,
) -> None:
    """A ``#[path]``-loaded file resolves its children from its own directory.

    Rust gives a path-loaded file the ``mod.rs`` directory rule: a ``mod
    child;`` inside it reads ``child.rs`` beside it, not from a directory
    named after its stem. Reading the other way would look for the child in a
    directory Rust never uses and report its proof unreachable.
    """
    crate_root = _write_package(tmp_path, '#[path = "elsewhere.rs"]\nmod proofs;\n')
    (crate_root.parent / "elsewhere.rs").write_text(
        "mod nested;\n", encoding="utf-8"
    )
    (crate_root.parent / "nested.rs").write_text(
        "#[kani::proof]\nfn verify_nested() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == ["proofs::nested::verify_nested"], (
        f"expected the child beside the path-loaded file, got {_names(inventory)!r}"
    )
    assert _kinds(inventory) == [], f"expected no reports, got {inventory.unresolved!r}"


def test_reader_applies_an_inline_module_path_attribute(
    tmp_path: Path,
) -> None:
    """An inline module's ``#[path]`` renames the directory its children use.

    For ``#[path = "thread_files"] mod thread { #[path = "tls.rs"] mod
    local_data; }`` Rust loads ``thread_files/tls.rs``, while the harness
    identity stays ``thread::local_data::...``. Reading the child from
    ``thread/tls.rs`` would miss a valid proof or read the wrong file.
    """
    (_write_package(tmp_path, ""))
    crate_root = _write_package(
        tmp_path,
        '#[path = "thread_files"]\nmod thread {\n    #[path = "tls.rs"]\n'
        "    mod local_data;\n}\n",
        name="other",
    )
    directory = crate_root.parent / "thread_files"
    directory.mkdir()
    (directory / "tls.rs").write_text(
        "#[kani::proof]\nfn verify_tls() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    names = _names(inventory)
    assert names == ["thread::local_data::verify_tls"], (
        f"the inline #[path] must name the child's directory, got {names!r}"
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


def test_reader_reads_a_raw_string_path_attribute(tmp_path: Path) -> None:
    """Rust accepts a raw string for a ``#[path]`` value."""
    crate_root = _write_package(
        tmp_path, '#[path = r"elsewhere.rs"] mod proofs;\n'
    )
    (crate_root.parent / "elsewhere.rs").write_text(
        "#[kani::proof]\nfn verify_raw() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_raw"], (
        f"a raw-string #[path] must be read, got {names!r}"
    )


def test_reader_reports_an_unreadable_path_value(tmp_path: Path) -> None:
    """A ``#[path]`` shape this reader cannot parse is reported, not guessed.

    Falling back to the default location would look for the module where the
    attribute does not point, so a proof that moved would vanish silently.
    """
    _write_package(tmp_path, '#[path = concat!("a", ".rs")] mod proofs;\n')
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["path-attribute"], (
        f"an unreadable #[path] value must be reported, got {kinds!r}"
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


def test_reader_reads_a_multiline_path_attribute(tmp_path: Path) -> None:
    """A ``#[path]`` spread over lines still names the file it wraps.

    Rust permits the value to sit on its own line. A reader that stopped at the
    closing bracket's line would lose the value and treat an ordinary module as
    if it had no path at all.
    """
    crate_root = _write_package(
        tmp_path,
        '#[path =\n    "elsewhere.rs"\n]\nmod proofs;\n',
    )
    (crate_root.parent / "elsewhere.rs").write_text(
        "#[kani::proof]\nfn verify_moved() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_moved"], (
        f"the multiline #[path] must be read, got {names!r}"
    )


def test_reader_finds_a_module_declared_after_a_same_line_attribute(
    tmp_path: Path,
) -> None:
    """``#[path = "elsewhere.rs"] mod proofs;`` declares ``proofs``.

    The attribute and the declaration share a line, so a pattern anchored at
    the line start finds neither. The module is ordinary Rust and its proof
    must be discovered.
    """
    crate_root = _write_package(
        tmp_path, '#[path = "elsewhere.rs"] mod proofs;\n'
    )
    (crate_root.parent / "elsewhere.rs").write_text(
        "#[kani::proof]\nfn verify_same_line() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_same_line"], (
        f"a same-line attribute must not hide the declaration, got {names!r}"
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


def test_reader_does_not_carry_an_attribute_past_an_item_boundary(
    tmp_path: Path,
) -> None:
    """A ``#[path]`` belongs to the declaration it precedes, not the next one.

    ``#[path = "one.rs"] mod first;`` followed by ``pub mod second;`` gives
    ``first`` the file and leaves ``second`` without one. A scan that starts
    after ``second``'s visibility qualifier and walks back past the semicolon
    attaches the earlier item's attribute to it, so a valid tree would fail
    the manifest contract.
    """
    crate_root = _write_package(
        tmp_path,
        '#[path = "one.rs"] mod first;\npub mod second;\n',
    )
    directory = crate_root.parent
    (directory / "one.rs").write_text("", encoding="utf-8")
    (directory / "second.rs").write_text("", encoding="utf-8")
    inventory = discover(tmp_path)
    kinds = _kinds(inventory)
    assert kinds == [], (
        f"both declarations resolve, so nothing may be reported, got {kinds!r}"
    )


def test_reader_reports_a_proof_below_a_repeated_proof_free_module(
    tmp_path: Path,
) -> None:
    """A shared parent's child keeps one identity, so the second is reported.

    Two declarations share a ``#[path]`` to a file holding no proof of its
    own, but that file's child module holds one. The child is read once, under
    the first path, so its second identity is lost; the repeat is reported
    rather than silently dropping it. Sharing a parent whose subtree holds no
    proof stays accepted, which the negative control below covers.
    """
    crate_root = _write_package(
        tmp_path,
        '#[path = "shared.rs"] mod first;\n#[path = "shared.rs"] mod second;\n',
    )
    # A `#[path]`-loaded file resolves its own children from its directory,
    # so `shared.rs`'s `mod proofs;` reads `src/proofs.rs`.
    (crate_root.parent / "shared.rs").write_text(
        "mod proofs;\n", encoding="utf-8"
    )
    (crate_root.parent / "proofs.rs").write_text(
        "#[kani::proof]\nfn verify_shared() {}\n", encoding="utf-8"
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["repeated-source"], (
        f"a lost identity below a repeated parent must be reported, got {kinds!r}"
    )


def test_reader_accepts_a_repeated_proof_free_subtree(tmp_path: Path) -> None:
    """The negative control for the repeat above: no proof, no report.

    Sharing a test-only source between two ``#[path]`` modules is ordinary
    Rust, and a subtree that declares no harness loses no identity when it is
    read once.
    """
    crate_root = _write_package(
        tmp_path,
        '#[path = "shared.rs"] mod first;\n#[path = "shared.rs"] mod second;\n',
    )
    (crate_root.parent / "shared.rs").write_text(
        "mod helper;\n", encoding="utf-8"
    )
    (crate_root.parent / "helper.rs").write_text(
        "pub fn work() -> bool { true }\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _kinds(inventory) == [], (
        f"a proof-free repeat must stay accepted, got {inventory.unresolved!r}"
    )


def test_reader_terminates_on_a_self_referential_path_module(
    tmp_path: Path,
) -> None:
    """A shared module that declares itself is Rust, and must not loop.

    Two declarations share a file whose only child is itself, so the module
    graph is cyclic. The repeat walk keeps its own visited set, because the
    module path grows on each visit and would otherwise never repeat.
    """
    crate_root = _write_package(
        tmp_path,
        '#[path = "loop.rs"] mod first;\n#[path = "loop.rs"] mod second;\n',
    )
    (crate_root.parent / "loop.rs").write_text(
        '#[path = "loop.rs"] mod inner;\n', encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"a self-referential module declares no proof, got {inventory.proofs!r}"
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