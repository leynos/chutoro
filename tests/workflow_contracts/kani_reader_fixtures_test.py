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

import typing as typ
from pathlib import Path

import pytest

from kani_harness_reading import UnresolvedDeclaration, discover
from kani_manifest_test import _validate_entry
from kani_reader_support import _kinds, _names


@pytest.mark.parametrize(
    ("what", "source"),
    (
        ("comment", "// #[kani::proof]\n// fn verify_commented() {}\n"),
        (
            "string literal",
            'const DOC: &str = "#[kani::proof]\\nfn verify_quoted() {}";\n',
        ),
    ),
)


def test_reader_ignores_a_proof_that_is_not_code(
    tmp_path: Path, write_package: typ.Callable[..., Path], what: str, source: str
) -> None:
    """A proof written in a comment or a string is text, not a declaration."""
    write_package(source)
    names = _names(discover(tmp_path))
    assert names == [], f"a {what} must not declare a proof, got {names!r}"


def test_reader_derives_a_nested_inline_module_path(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A proof's harness path includes every inline module enclosing it."""
    write_package("mod outer {\n    mod inner {\n        #[kani::proof]\n"
        "        fn verify_nested() {}\n    }\n}\n",
    )
    names = _names(discover(tmp_path))
    assert names == ["outer::inner::verify_nested"], (
        f"the nested inline path must be derived, got {names!r}"
    )


@pytest.mark.parametrize(
    ("declaration", "expected"),
    (
        ("mod proofs;\n", "proofs::verify_file"),
        ("mod\n    proofs\n;\n", "proofs::verify_split"),
    ),
    ids=("one line", "split over lines"),
)
def test_reader_derives_a_file_based_module_path(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
    declaration: str,
    expected: str,
) -> None:
    """A ``mod name;`` declaration contributes its module segment.

    Rust permits whitespace, including a newline, between the keyword and the
    name, so both spellings must resolve to the same child file.
    """
    crate_root = write_package(declaration)
    proof = f"#[kani::proof]\nfn {expected.split('::')[-1]}() {{}}\n"
    (crate_root.parent / "proofs.rs").write_text(proof, encoding="utf-8")
    names = _names(discover(tmp_path))
    assert names == [expected], f"the module path must be derived, got {names!r}"


def test_reader_reports_a_cfg_attr_wrapped_proof(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A proof whose existence depends on configuration cannot be classified."""
    write_package('#[cfg_attr(kani, kani::proof)]\nfn verify_conditional() {}\n',
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["cfg-attr"], (
        f"a cfg_attr-wrapped proof must be reported, got {kinds!r}"
    )


def test_reader_reports_a_proof_inside_a_macro(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A proof that only exists after expansion cannot be attributed."""
    write_package("macro_rules! declare {\n    () => {\n        #[kani::proof]\n"
        "        fn verify_expanded() {}\n    };\n}\n",
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["macro"], (
        f"a proof inside a macro must be reported, got {kinds!r}"
    )


def test_reader_reports_a_proof_bearing_source_no_module_reaches(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """An orphaned proof file is reported rather than silently dropped.

    Kani compiles a module only when the crate declares it, so a proof in an
    unreferenced file would never run. Reporting it keeps the manifest from
    promising a result the build cannot produce.
    """
    crate_root = write_package("")
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


def test_reader_accepts_a_root_level_proof_identity(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A proof at a crate's root has no module segment.

    Its harness path is the bare function name, so manifest validation must
    accept a name without ``::`` as well as one with it; requiring a module
    would reject the identity the reader reports and the comparison could
    never agree for such a package.
    """
    names = _names(
        discover(write_package("#[kani::proof]\nfn verify_root() {}\n").parent.parent)
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


def test_reader_reads_a_proof_after_an_inline_module_opener(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """``mod proofs { #[kani::proof]`` puts the attribute on the opener's line.

    Rust gives no meaning to the line break before an item, so a reader that
    worked line by line would stop at ``mod`` and lose the proof.
    """
    write_package("mod proofs { #[kani::proof]\n    fn verify_inline() {}\n}\n",
    )
    names = _names(discover(tmp_path))
    assert names == ["proofs::verify_inline"], (
        f"an attribute on the module opener's line must be read, got {names!r}"
    )


def test_reader_keeps_a_module_file_without_a_proof_out_of_the_inventory(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """A reachable module holding no proof adds nothing to the inventory.

    The negative control for module traversal: it must not invent a
    declaration for a file it read but found nothing in.
    """
    crate_root = write_package("mod helper;\n")
    (crate_root.parent / "helper.rs").write_text(
        "pub fn work() -> bool { true }\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    names = _names(inventory)
    assert names == [], f"a helper module holds no proof, got {names!r}"
    kinds = _kinds(inventory)
    assert kinds == [], f"nothing must be reported, got {kinds!r}"


def test_reader_reads_two_attributes_sharing_a_line(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """``#[cfg(kani)] #[kani::proof]`` on one line still declares a proof.

    Rust gives no meaning to the line break between two outer attributes, so
    the two spellings are one declaration. Reading the line as a single
    attribute sees only ``cfg`` and loses the proof.
    """
    write_package("#[cfg(kani)] #[kani::proof]\nfn verify_same_line() {}\n",
    )
    names = _names(discover(tmp_path))
    assert names == ["verify_same_line"], (
        f"two attributes on one line must both be read, got {names!r}"
    )


def test_reader_reports_an_included_source(
    tmp_path: Path, write_package: typ.Callable[..., Path]
) -> None:
    """A literal ``include!`` compiles a file this walk never follows.

    Following only ``mod`` declarations would omit its proofs without a trace,
    so the include is reported and the inventory fails closed.
    """
    crate_root = write_package('include!("../proofs.rs");\n')
    (crate_root.parent.parent / "proofs.rs").write_text(
        "#[kani::proof]\nfn verify_included() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _kinds(inventory) == ["include"], (
        f"an include! must be reported, got {inventory.unresolved!r}"
    )


def test_reader_reports_a_proof_in_a_parenthesised_macro(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A macro body may be delimited by parentheses or brackets, not only braces."""
    write_package("macro_rules! declare (\n    () => {\n        #[kani::proof]\n"
        "        fn verify() {}\n    };\n);\n",
    )
    kinds = _kinds(discover(tmp_path))
    assert kinds == ["macro"], (
        f"a proof inside a parenthesised macro must be reported, got {kinds!r}"
    )


def test_unresolved_report_names_the_file_and_line(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """A report without a location would not be actionable in review."""
    write_package("// filler\nfn other() {}\n#[cfg_attr(kani, kani::proof)]\nfn verify_conditional() {}\n",
    )
    (report,) = discover(tmp_path).unresolved
    assert isinstance(report, UnresolvedDeclaration), (
        f"expected an unresolved report, got {report!r}"
    )
    assert report.line == 4, f"expected the declaration's line, got {report!r}"
    assert report.path.endswith("src/lib.rs"), (
        f"expected a source path, got {report!r}"
    )
