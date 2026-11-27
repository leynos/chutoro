"""Walk each crate's module tree and attribute its proofs.

The workflow-contract suite needs the whole proof surface, not merely the
files whose names match ``kani_*``. Six proofs live in an inline
``#[cfg(kani)] mod kani_proofs`` block at the bottom of
``chutoro-core/src/distance/mod.rs``, and a filename-based inventory never saw
them. This module therefore walks each crate's module tree and reads
declarations out of the source.

It reuses :mod:`kani_source_reading` for the per-file reading and
:mod:`kani_module_paths` for child resolution. Its own work is the file-based
link: a module reached through ``mod name;`` contributes its declaration's
name, and a ``#[path]`` that names no readable file is reported rather than
skipped.

Anything this reader cannot attribute to a module path is reported rather than
skipped: an unresolvable ``#[path]`` module, a proof wrapped in ``cfg_attr``, a
proof inside a macro definition, and a proof-bearing file no ``mod``
declaration reaches are all unsupported shapes, and the contract turns each
report into a failure. Failing closed is the point; a reader that guessed
would let a proof disappear from the inventory without anything noticing.

Re-use policy: import from :mod:`kani_manifest_test` and the rewritten
:mod:`kani_fast_tier_test` only. Do not grow a second Rust scanner beside it.
"""

from __future__ import annotations

import dataclasses
import typing as typ
from pathlib import Path

from cargo_packages_reading import _Package, _is_package_tree, _packages
from kani_inventory_model import Inventory, ProofDeclaration, UnresolvedDeclaration
from kani_module_paths import _Context
from kani_module_walk import _Reader
from kani_source_sweep import _unreached
from workflow_support import ROOT


def discover(root: Path = ROOT) -> Inventory:
    """Return every proof declaration under `root` and every gap in reading it.

    Parameters
    ----------
    root
        The repository tree to read. Defaults to this checkout.

    Examples
    --------
    >>> inventory = discover()
    >>> len(inventory.proofs) >= 17
    True
    >>> any(
    ...     proof.pretty_name == "distance::kani_proofs::verify_cosine_bounded_3d"
    ...     for proof in inventory.proofs
    ... )
    True

    Returns
    -------
    Inventory
        The attributable declarations and the constructs that could not be
        attributed.
    """
    root = Path(root).resolve()
    proofs: list[ProofDeclaration] = []
    unresolved: list[UnresolvedDeclaration] = []
    by_package: dict[str, list[_Package]] = {}
    for package in _packages(root):
        by_package.setdefault(package.name, []).append(package)
    for name, roots in by_package.items():
        reader = _Reader(name, root)
        for package in roots:
            reader.visit(
                package.root,
                _Context((), (), package.is_binary),
                is_crate_root=True,
            )
        proofs.extend(reader.proofs)
        unresolved.extend(reader.unresolved)
        for path in _unreached(roots, reader.visited):
            unresolved.append(
                UnresolvedDeclaration(
                    kind="unreachable",
                    path=path.relative_to(root).as_posix(),
                    line=1,
                    detail=(
                        "proof-bearing source is not reached by any module "
                        "declaration from the crate root"
                    ),
                )
            )
    proofs.sort(key=lambda proof: (proof.package, proof.pretty_name))
    unresolved.sort(key=lambda report: (report.path, report.line, report.kind))
    return Inventory(proofs=tuple(proofs), unresolved=tuple(unresolved))