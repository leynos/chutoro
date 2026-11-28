"""Configuration-guard fixtures for the Kani source reader.

The gate compiles the library without ``cfg(test)`` and with ``kani`` set, so a
``cfg`` predicate can exclude a proof from the build it is meant to verify.
These cases spell the guards that matter — on the function, on an inline
module, on a file module, on the file itself — and the ones that do not,
because evaluating them is the only way to tell the difference. The other
reader cases live in :mod:`kani_reader_fixtures_test`.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ
from pathlib import Path

import pytest

from kani_harness_reading import discover
from kani_reader_support import _kinds, _names, _write_package


def test_reader_reports_a_proof_behind_a_test_guard(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """The gate compiles without ``cfg(test)``, so such a proof cannot run.

    Recording it as a harness would let the manifest require something the
    recipe can never verify.
    """
    write_package("#[cfg(all(kani, test))]\n#[kani::proof]\nfn verify_test_only() {}\n",
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"a test-gated proof must not be a harness, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == ["cfg-test"], (
        f"a test-gated proof must be reported, got {inventory.unresolved!r}"
    )


def test_reader_carries_a_modules_guard_into_its_children(
    tmp_path: Path,
) -> None:
    """A test guard on a module covers every proof below it.

    ``cargo kani`` does not enable ``cfg(test)``, so the module is not
    compiled at all in the gate; a proof inside it cannot run even when the
    function carries no guard of its own.
    """
    crate_root = _write_package(
        tmp_path, "#[cfg(all(kani, test))]\nmod proofs;\n"
    )
    (crate_root.parent / "proofs.rs").write_text(
        "#[kani::proof]\nfn verify_child() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"a proof under a test-gated module must not be a harness, got {inventory.proofs!r}"
    )
    kinds = _kinds(inventory)
    assert kinds == ["cfg-test"], (
        f"the proof must be reported as test-gated, got {kinds!r}"
    )


@pytest.mark.parametrize(
    ("what", "guard"),
    (
        ("a file's inner guard", "#![cfg(test)]\n"),
        ("a compound test guard", '#[cfg(all(not(feature = "opt"), test))]\n'),
    ),
)


def test_reader_reports_every_shape_of_test_guard(
    tmp_path: Path, what: str, guard: str
) -> None:
    """A guard the gate's configuration excludes is reported, whatever it contains.

    The predicate is evaluated, not searched: `all(not(feature), test)` is
    still false without `cfg(test)`, and its `test` follows the first `)`.
    """
    _write_package(
        tmp_path, f'{guard}#[kani::proof]\nfn verify_guarded() {{}}\n'
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"{what} must not certify a harness, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == ["cfg-test"], (
        f"{what} must be reported, got {inventory.unresolved!r}"
    )


def test_reader_honours_an_inline_modules_inner_guard(
    tmp_path: Path, write_package: typ.Callable[..., Path]
) -> None:
    """``mod p { #![cfg(test)] .. }`` gates everything inside the module."""
    write_package(
        "mod proofs {\n    #![cfg(test)]\n    #[kani::proof]\n"
        "    fn verify_inner() {}\n}\n"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"an inner-guarded proof must not be a harness, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == ["cfg-test"], (
        f"the proof must be reported as test-gated, got {inventory.unresolved!r}"
    )


@pytest.mark.parametrize(
    ("what", "guard"),
    (
        ("a negated test guard", "#[cfg(not(test))]\n"),
        ("an any() guard naming kani", "#[cfg(any(kani, test))]\n"),
    ),
)


def test_reader_keeps_a_proof_the_gate_does_compile(
    tmp_path: Path, what: str, guard: str
) -> None:
    """The negative control: these guards are true without `cfg(test)`.

    `not(test)` is true in the gate's build, and `any(kani, test)` is true
    because Kani sets `kani`; both proofs run and must be inventoried.
    """
    _write_package(
        tmp_path, f'{guard}#[kani::proof]\nfn verify_compiled() {{}}\n'
    )
    names = _names(discover(tmp_path))
    assert names == ["verify_compiled"], (
        f"{what} must keep the proof, got {names!r}"
    )


def test_reader_carries_an_inline_guard_across_a_file_module(
    tmp_path: Path,
    write_package: typ.Callable[..., Path],
) -> None:
    """An enclosing inline module's test guard covers a file module's proof.

    The child is reached through `outer`, so Rust excludes it from the gate
    build even though `child.rs` carries no guard of its own.
    """
    crate_root = write_package(
        "#[cfg(all(kani, test))]\nmod outer {\n    mod child;\n}\n"
    )
    (crate_root.parent / "outer").mkdir()
    (crate_root.parent / "outer" / "child.rs").write_text(
        "#[kani::proof]\nfn verify_child() {}\n", encoding="utf-8"
    )
    inventory = discover(tmp_path)
    assert _names(inventory) == [], (
        f"a proof under a gated inline module must not be a harness, got {inventory.proofs!r}"
    )
    assert _kinds(inventory) == ["cfg-test"], (
        f"the proof must be reported as test-gated, got {inventory.unresolved!r}"
    )


def test_reader_keeps_a_proof_behind_a_kani_guard(tmp_path: Path, write_package: typ.Callable[..., Path]) -> None:
    """The negative control: ``#[cfg(kani)]`` is the gate's own configuration."""
    write_package("#[cfg(kani)]\n#[kani::proof]\nfn verify_gated() {}\n",
    )
    names = _names(discover(tmp_path))
    assert names == ["verify_gated"], f"a kani-gated proof must be read, got {names!r}"
