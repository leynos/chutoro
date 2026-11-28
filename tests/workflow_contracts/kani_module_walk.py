"""Walk one package's module tree and record what it reaches.

The walker owns the traversal state: which files it has read under which module
path, the declarations it has attributed, and the constructs it could not
attribute. Reading a single file's attributes belongs to
:mod:`kani_source_reading`; resolving a child's location belongs to
:mod:`kani_module_paths`; finding sources no declaration reaches belongs to
:mod:`kani_source_sweep`.

Re-use policy: import from :mod:`kani_harness_reading` only.
"""

from __future__ import annotations

import re
import typing as typ
from pathlib import Path

from kani_attribute_reading import _line_of, _proof_attributes, _proof_kind
from kani_cfg_reading import gated_by_test
from kani_inventory_model import ProofDeclaration, UnresolvedDeclaration
from kani_module_paths import _Context, _Location, _child_file, _is_mod_rs
from kani_source_reading import (
    _ModuleItem,
    _Source,
    _inside,
    _matching_delimiter,
    _module_chain,
    _parse,
)
from kani_source_sweep import _declared_children, _holds_proof
from rust_source_reading import _FUNCTION


#: Rust compiles an included file's items, but this walk follows only ``mod``
#: declarations, so every include is reported. The scan runs on the
#: literal-blanked text, so a mention inside a string is not one.
_INCLUDE = re.compile(r"\binclude\s*!\s*(?P<opening>[({[])")



class _Reader:
    """Reads one package's module tree and records what it finds."""

    def __init__(self, package: str, repo_root: Path) -> None:
        """Start a sweep of one package under `repo_root`."""
        self._package = package
        self._repo_root = repo_root
        self._visited: dict[Path, tuple[str, ...]] = {}
        self.proofs: list[ProofDeclaration] = []
        self.unresolved: list[UnresolvedDeclaration] = []

    @property
    def visited(self) -> frozenset[Path]:
        """Return the resolved paths this sweep read."""
        return frozenset(self._visited)

    def _report(self, kind: str, path: str, line: int, detail: str) -> None:
        """Record one construct that could not be attributed."""
        self.unresolved.append(
            UnresolvedDeclaration(kind=kind, path=path, line=line, detail=detail)
        )

    def visit(
        self,
        file: Path,
        context: _Context = _Context((), (), False),
        *,
        is_crate_root: bool = False,
        via_path: bool = False,
    ) -> None:
        """Read `file` as the module `module`, following its declarations.

        `context` carries the module path, the ``#[cfg]`` attributes of the
        declarations that reached this file, and whether it belongs to a
        binary target. A module gated by a test guard puts every proof below it
        in that configuration too, and ``--lib`` never compiles a binary.

        A file reached twice under the same module path is a revisited module
        and is skipped. A file reached twice under *different* paths is not:
        Rust admits it when two declarations share a ``#[path]`` target, and
        each path yields its own harness identities. Dropping the second would
        silently under-report the inventory, so it is reported instead.
        """
        resolved = file.resolve()
        location = _Location(
            file=file, is_crate_root=is_crate_root, via_path=via_path
        )
        if resolved in self._visited:
            self._note_revisit(resolved, context, location)
            return
        self._visited[resolved] = context.module
        source = _parse(file)
        relative = file.relative_to(self._repo_root).as_posix()
        if source is None:
            # A file the sweep reached but could not decode is not the same as
            # one with no declarations: skipping it would shrink the inventory
            # silently, which is exactly the loss this contract exists to catch.
            self._report(
                "unreadable",
                relative,
                1,
                "the module resolves to this source, but it cannot be read as "
                "UTF-8, so its declarations cannot be checked",
            )
            return
        self._report_includes(source, relative)
        if source.inner:
            context = _Context(
                context.module,
                (*context.guards, *source.inner),
                context.binary,
            )
        self._record_proofs(source, relative, context)
        self._follow(location, source, context)

    def _note_revisit(
        self, resolved: Path, context: _Context, location: _Location
    ) -> None:
        """Report a source reached again under a different module path.

        Only a proof-bearing file can lose an identity by being read once.
        Sharing a test-only source between two ``#[path]`` modules is ordinary
        Rust and declares no harness, so reporting it would fail the contract
        on a valid tree.

        A proof-free file can still reach one, though: it may declare a child
        module, and that child's harnesses are read only under the first path.
        A repeat is therefore reported when either the file or a descendant of
        it declares a proof.
        """
        first = self._visited[resolved]
        if first == context.module:
            return
        source = _parse(location.file)
        if source is None:
            return
        held = _holds_proof(source) or bool(
            self._descendant_proofs(location, source, context)
        )
        if not held:
            return
        self._report(
            "repeated-source",
            location.file.relative_to(self._repo_root).as_posix(),
            1,
            "the same source is reached as both "
            f"{'::'.join(first)} and {'::'.join(context.module)}, so a harness "
            "identity below it would be dropped",
        )

    def _descendant_proofs(
        self, location: _Location, source: _Source, context: _Context
    ) -> list[str]:
        """Return the harness paths a repeated file's subtree would declare.

        The walk keeps its own set of resolved files rather than using
        :meth:`visit`, because it must terminate on the cyclic ``#[path]``
        graph Rust admits: a file that reaches itself would otherwise be
        followed forever under a module path that grows by one segment each
        time.
        """
        found: list[str] = []
        seen = {location.file.resolve()}
        pending = [(location, source, context.module)]
        while pending:
            current, current_source, current_module = pending.pop()
            for _, child, child_module in _declared_children(
                current, current_source, current_module
            ):
                if child is None or child.file.resolve() in seen:
                    continue
                seen.add(child.file.resolve())
                child_source = _parse(child.file)
                if child_source is None:
                    continue
                if _holds_proof(child_source):
                    found.append("::".join(child_module))
                    continue
                pending.append((child, child_source, child_module))
        return found

    def _record_proofs(
        self, source: _Source, relative: str, context: _Context
    ) -> None:
        """Record each proof in one file, or report why it cannot be."""
        for match in _FUNCTION.finditer(source.code):
            attributes = _proof_attributes(source.code, match.start())
            kind = _proof_kind(attributes)
            if kind is None:
                continue
            line = _line_of(source.text, match.start())
            if kind == "cfg-attr":
                self._report(
                    "cfg-attr",
                    relative,
                    line,
                    "proof is wrapped in cfg_attr, so whether it is a "
                    "declaration depends on configuration",
                )
                continue
            if _inside(match.start(), source.macros):
                self._report(
                    "macro",
                    relative,
                    line,
                    "proof is declared inside a macro definition",
                )
                continue
            chain = _module_chain(source.modules, match.start())
            # The gate compiles the library without `cfg(test)`, so a proof
            # behind a test guard exists in the source but never runs. Counting
            # it as a harness would let the manifest require something the
            # recipe cannot verify.
            guards = [
                *context.guards,
                *attributes,
                *(attr for item in chain for attr in item.attributes),
            ]
            if gated_by_test(guards):
                self._report(
                    "cfg-test",
                    relative,
                    line,
                    "proof is behind a cfg(test) guard, which the Kani gate "
                    "does not enable, so it cannot run",
                )
                continue
            # The gate compiles with `--lib`, so a binary target's harnesses
            # never run in either tier. Recording one would let a mandatory
            # manifest entry be satisfied by a proof the run cannot execute.
            if context.binary:
                self._report(
                    "binary-target",
                    relative,
                    line,
                    "proof is declared in a binary target, which the Kani gate "
                    "does not compile",
                )
                continue
            pretty = "::".join(
                (*context.module, *(item.name for item in chain), match["name"])
            )
            self.proofs.append(
                ProofDeclaration(
                    package=self._package,
                    pretty_name=pretty,
                    path=relative,
                    line=line,
                )
            )

    def _follow(
        self, location: _Location, source: _Source, context: _Context
    ) -> None:
        """Follow each file-based module declaration in one file."""
        self._report_inline_paths(location, source)
        for item, child, child_module in _declared_children(
            location, source, context.module
        ):
            if child is None:
                self._report_missing_path(location, source, item)
                continue
            chain = _module_chain(source.modules, item.start)
            self.visit(
                child.file,
                _Context(
                    child_module,
                    (
                        *context.guards,
                        *(attr for parent in chain for attr in parent.attributes),
                        *item.attributes,
                    ),
                    context.binary,
                ),
                via_path=child.via_path,
            )

    def _report_includes(self, source: _Source, relative: str) -> None:
        """Report each ``include!`` the module walk cannot follow.

        Rust compiles the named file's items into this module, so a proof
        there runs under the gate; following only ``mod`` declarations would
        omit it without a trace.
        """
        for found in _INCLUDE.finditer(source.code):
            opening = found.end() - 1
            closing = _matching_delimiter(source.code, opening, found["opening"])
            if closing is None:
                closing = len(source.code)
            argument = source.declarations[opening + 1 : closing].strip()
            self._report(
                "include",
                relative,
                _line_of(source.text, found.start()),
                f"the module includes {argument!r} with include!, which this "
                "reader cannot follow",
            )

    def _report_inline_paths(self, location: _Location, source: _Source) -> None:
        """Report an inline module whose own ``#[path]`` cannot be read.

        That value renames the directory its children use, so a child resolved
        without it would be looked for where Rust may not put it.
        """
        for item in source.modules:
            if item.inline and item.path_unreadable:
                self._report_missing_path(location, source, item)

    def _report_missing_path(
        self, location: _Location, source: _Source, item: _ModuleItem
    ) -> None:
        """Report a ``#[path]`` declaration this reader cannot follow.

        A plain ``mod name;`` that names no file is Rust's own compile error,
        so only a ``#[path]`` is this reader's to explain. A value in a shape
        the reader cannot read is reported too: replacing it with the default
        rule would look for the module where the attribute does not point.
        """
        if item.path_unreadable:
            self._report(
                "path-attribute",
                location.file.relative_to(self._repo_root).as_posix(),
                _line_of(source.text, item.start),
                "the #[path] value is in a form this reader cannot read, so "
                "the module's file cannot be located",
            )
            return
        if item.path_value is None:
            return
        self._report(
            "path-attribute",
            location.file.relative_to(self._repo_root).as_posix(),
            _line_of(source.text, item.start),
            f"#[path = {item.path_value!r}] does not resolve to a readable "
            "source file",
        )
