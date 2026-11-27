"""Path-attribute fixtures for the Kani source reader.

Every case here spells a ``#[path]`` in some shape Rust accepts, or in a shape
the reader cannot read, and asserts what :func:`discover` makes of it: a value
that resolves, one that does not, a raw string, a conditional assignment, a
mention inside documentation, and an inline module that renames its children's
directory. The general reader cases live in
:mod:`kani_reader_fixtures_test`.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ
from pathlib import Path

import pytest

from kani_harness_reading import discover
from kani_reader_support import _kinds, _names, _write_package


def test_reader_reports_an_unresolvable_path_attribute(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A ``#[path]`` naming no file is reported rather than skipped."""
    write_package('#[path = "absent.rs"]\nmod proofs;\n')
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["path-attribute"], (
        f"an unresolvable #[path] must be reported, got {kinds!r}"
    )


def test_reader_follows_a_resolvable_path_attribute(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A ``#[path]`` naming a real file still contributes its module's name.

    Rust takes the module's name from the declaration, not from the file's
    location, so a resolvable path attribute leaves the harness identity
    derivable. Reporting these instead would fail the contract on the
    repository's ordinary test-only ``#[path]`` modules.
    """
    crate_root = write_package('#[path = "elsewhere.rs"]\nmod proofs;\n')
    (crate_root.parent / "elsewhere.rs").write_text(
        "#[kani::proof]\nfn verify_moved() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_moved"], (
        f"a resolvable #[path] must contribute its module name, got {names!r}"
    )


def test_reader_resolves_a_path_loaded_module_children_beside_it(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """A ``#[path]``-loaded file resolves its children from its own directory.

    Rust gives a path-loaded file the ``mod.rs`` directory rule: a ``mod
    child;`` inside it reads ``child.rs`` beside it, not from a directory
    named after its stem. Reading the other way would look for the child in a
    directory Rust never uses and report its proof unreachable.
    """
    crate_root = write_package('#[path = "elsewhere.rs"]\nmod proofs;\n')
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
    write_package: typ.Callable[..., Path],
) -> None:
    """An inline module's ``#[path]`` renames the directory its children use.

    For ``#[path = "thread_files"] mod thread { #[path = "tls.rs"] mod
    local_data; }`` Rust loads ``thread_files/tls.rs``, while the harness
    identity stays ``thread::local_data::...``. Reading the child from
    ``thread/tls.rs`` would miss a valid proof or read the wrong file.
    """
    (write_package(""))
    crate_root = write_package('#[path = "thread_files"]\nmod thread {\n    #[path = "tls.rs"]\n'
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


def test_reader_reports_an_unreadable_path_value(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A ``#[path]`` shape this reader cannot parse is reported, not guessed.

    Falling back to the default location would look for the module where the
    attribute does not point, so a proof that moved would vanish silently.
    """
    write_package('#[path = concat!("a", ".rs")] mod proofs;\n')
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["path-attribute"], (
        f"an unreadable #[path] value must be reported, got {kinds!r}"
    )


@pytest.mark.parametrize(
    ("what", "declaration"),
    (
        ("a raw string", '#[path = r"elsewhere.rs"] mod proofs;\n'),
        ("a value on its own line", '#[path =\n    "elsewhere.rs"\n]\nmod proofs;\n'),
        (
            "an attribute sharing the declaration's line",
            '#[path = "elsewhere.rs"] mod proofs;\n',
        ),
    ),
)
def test_reader_reads_a_path_in_every_spelling(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
    what: str,
    declaration: str,
) -> None:
    """Rust accepts every one of these spellings for one ``#[path]``.

    The value may be a raw string, may sit on its own line, and may share the
    declaration's line; a reader that missed any spelling would look for the
    module in the default location and lose the proof.
    """
    crate_root = write_package(declaration)
    (crate_root.parent / "elsewhere.rs").write_text(
        "#[kani::proof]\nfn verify_moved() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_moved"], (
        f"{what} must be read as a #[path], got {names!r}"
    )


def test_reader_does_not_carry_an_attribute_past_an_item_boundary(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """A ``#[path]`` belongs to the declaration it precedes, not the next one.

    ``#[path = "one.rs"] mod first;`` followed by ``pub mod second;`` gives
    ``first`` the file and leaves ``second`` without one. A scan that starts
    after ``second``'s visibility qualifier and walks back past the semicolon
    attaches the earlier item's attribute to it, so a valid tree would fail
    the manifest contract.
    """
    crate_root = write_package('#[path = "one.rs"] mod first;\npub mod second;\n',
    )
    directory = crate_root.parent
    (directory / "one.rs").write_text("", encoding="utf-8")
    # The proof in the second destination is what makes this test decisive: if
    # the reader skipped `second` or resolved it to `one.rs`, the identity
    # below would be missing even though nothing was reported.
    (directory / "second.rs").write_text(
        "#[kani::proof]\nfn verify_second() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == ["second::verify_second"], (
        f"the second declaration must read its own file, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == [], (
        f"both declarations resolve, so nothing may be reported, got {inventory.unresolved!r}"
    )


def test_reader_reports_a_proof_below_a_repeated_proof_free_module(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """A shared parent's child keeps one identity, so the second is reported.

    Two declarations share a ``#[path]`` to a file holding no proof of its
    own, but that file's child module holds one. The child is read once, under
    the first path, so its second identity is lost; the repeat is reported
    rather than silently dropping it. Sharing a parent whose subtree holds no
    proof stays accepted, which the negative control below covers.
    """
    crate_root = write_package('#[path = "shared.rs"] mod first;\n#[path = "shared.rs"] mod second;\n',
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


def test_reader_accepts_a_repeated_proof_free_subtree(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """The negative control for the repeat above: no proof, no report.

    Sharing a test-only source between two ``#[path]`` modules is ordinary
    Rust, and a subtree that declares no harness loses no identity when it is
    read once.
    """
    crate_root = write_package('#[path = "shared.rs"] mod first;\n#[path = "shared.rs"] mod second;\n',
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
    write_package: typ.Callable[..., Path],
) -> None:
    """A shared module that declares itself is Rust, and must not loop.

    Two declarations share a file whose only child is itself, so the module
    graph is cyclic. The repeat walk keeps its own visited set, because the
    module path grows on each visit and would otherwise never repeat.
    """
    crate_root = write_package('#[path = "loop.rs"] mod first;\n#[path = "loop.rs"] mod second;\n',
    )
    (crate_root.parent / "loop.rs").write_text(
        '#[path = "loop.rs"] mod inner;\n', encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"a self-referential module declares no proof, got {inventory.proofs!r}"
    )


def test_reader_reports_a_conditional_path_attribute(tmp_path: Path) -> None:
    """A ``cfg_attr`` path applies only under the named configuration.

    Whether the module's file moves depends on configuration this reader does
    not evaluate, so the declaration is reported rather than resolved by the
    default rule. The proof below sits outside ``src``, which is exactly where
    the default rule would never look.
    """
    directory = tmp_path / "conditional"
    (directory / "src").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "conditional"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (directory / "src" / "lib.rs").write_text(
        '#[cfg_attr(kani, path = "../proofs.rs")]\nmod proofs;\n',
        encoding="utf-8",
    )
    (directory / "proofs.rs").write_text(
        "#[kani::proof]\nfn verify_conditional() {}\n", encoding="utf-8"
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["path-attribute"], (
        f"a conditional #[path] must be reported, got {kinds!r}"
    )


def test_reader_ignores_a_path_mentioned_in_a_doc_attribute(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """Documentation may name a ``#[path]`` without declaring one.

    The doc string is an attribute's value, not an outer attribute of the
    declaration, so it neither provides a path nor makes the real module
    unresolvable.
    """
    crate_root = write_package('#[doc = r#"#[path = \\"absent.rs\\"]"#]\nmod proofs;\n',
    )
    (crate_root.parent / "proofs.rs").write_text(
        "#[kani::proof]\nfn verify_documented() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == ["proofs::verify_documented"], (
        f"a doc mention must not redirect the module, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == [], (
        f"a doc mention must not be reported, got {inventory.unresolved!r}"
    )


def test_reader_reads_a_path_after_a_doc_string_with_brackets(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """A bracket inside a doc string is text, not an attribute's bracket."""
    crate_root = write_package('#[doc = "use [square] brackets"]\n#[path = r"elsewhere.rs"]\nmod proofs;\n',
    )
    (crate_root.parent / "elsewhere.rs").write_text(
        "#[kani::proof]\nfn verify_bracketed() {}\n", encoding="utf-8"
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_bracketed"], (
        f"a doc bracket must not hide the path attribute, got {names!r}"
    )


def test_reader_ignores_a_path_inside_conditional_documentation(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """A ``cfg_attr`` that only documents a path assignment names no path."""
    write_package('#[cfg_attr(kani, doc = "example: path = old.rs")]\nmod proofs;\n',
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == [], (
        f"documentation must not read as a conditional path, got {kinds!r}"
    )


def test_reader_reports_an_unreadable_inline_module_path(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """An inline module's own ``#[path]`` decides its children's directory."""
    write_package('#[cfg_attr(kani, path = concat!("a", "b"))]\nmod outer {\n'
        "    mod proofs;\n}\n",
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["path-attribute"], (
        f"an unreadable inline path must be reported, got {kinds!r}"
    )
