"""Contract for the installer's default prefix and the PATH that reaches it.

``scripts/install-mold.sh`` installs to ``${HOME}/.local`` unless
``MOLD_INSTALL_PREFIX`` overrides it, and every other installer contract
overrides it. The Makefile's ``export PATH`` line is what makes that default
prefix usable: ``gcc`` resolves ``ld.mold`` from ``PATH``, so without the line
a developer who ran ``make install-dev-fast`` would still link with whichever
mold the host happened to provide.

Two contracts pin each half. The first installs with the default prefix and
links a real crate, proving the installed tree is one the linker route
accepts. The second is its control: with the ``PATH`` line removed and a
failing ``ld.mold`` ahead of the host's own, the same build must not reach the
installed binary. Without that control the first contract would pass on a
host that already had a working mold anywhere else.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from dev_fast_make_support import MAKEFILE_PATH, _skip_unless_pin_installed
from dev_fast_probe_support import (
    probe_environment,
    run_development_target,
    write_probe_repository,
)
from mold_install_support import Platform, Request, prepare

#: The line whose removal is the control. It is anchored on the whole line so
#: a reordered or reformatted assignment fails loudly rather than silently
#: mutating something else.
PATH_EXPORT = (
    "export PATH := $(HOME)/.cargo/bin:$(HOME)/.local/bin:"
    "$(HOME)/.bun/bin:$(PATH)"
)

#: An ``ld.mold`` that records every invocation and then fails. Placing it
#: ahead of the host's own is how the control tells "the PATH export found the
#: installed binary" from "something else on this host linked the crate".
POISON_SHIM = """#!/usr/bin/env bash
printf 'reached\\n' >> __LOG__
if [[ "${1:-}" == --version ]]; then
  printf 'mold 2.41.0\\n'
  exit 0
fi
exit 1
"""

def _mold_installable() -> bool:
    """Report whether this host can build and run the probe's Cranelift route."""
    return shutil.which("gcc") is not None


def _poisoned_path(directory: Path, log: Path) -> Path:
    """Write a failing ``ld.mold`` that records its calls, and return its dir."""
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "ld.mold"
    shim.write_text(POISON_SHIM.replace("__LOG__", str(log)), encoding="utf-8")
    shim.chmod(0o755)
    return directory


def test_the_installer_places_a_default_prefix_tree_the_linker_loads(
    tmp_path: Path,
) -> None:
    """``install-dev-fast`` must install somewhere the recipe can then find.

    The installer's default is ``${HOME}/.local`` and the recipe reaches that
    directory only through its ``PATH`` export. Running both against one
    private home is what joins those two halves: the install proves the
    default prefix is populated, and the build proves the exported ``PATH``
    is what the linker consults.
    """
    if not _mold_installable():
        pytest.skip("the linker route needs gcc, which this host does not provide")
    _skip_unless_pin_installed("the linker route links with the pinned nightly")
    installer = prepare(tmp_path, default_prefix=True, request=_linux_request())

    installed = installer.install()

    assert installed.returncode == 0, installed.stderr
    assert (installer.prefix / "bin" / "ld.mold").is_file(), (
        f"the default prefix must hold ld.mold: {installer.prefix}"
    )
    # A host may already carry a working mold, so a bare success would not
    # show that the installed one was the linker consulted. The failing shim
    # ahead of everything makes the success depend on the export: it pushes
    # `~/.local/bin` in front of the shim, so the build can only pass if the
    # export is what selected the installed binary.
    poison_log = tmp_path / "poison-log"
    directory = write_probe_repository(tmp_path / "probe")
    environment = probe_environment(directory, installer.environment.home)
    environment["PATH"] = (
        f"{_poisoned_path(tmp_path / 'poison', poison_log)}"
        f"{os.pathsep}{installer.environment.values['PATH']}"
    )

    built = run_development_target(directory, environment, "dev-build")

    assert built.returncode == 0, (
        "the recipe must link with the mold it installed under the default "
        f"prefix, ahead of the failing shim:\n{built.stdout}\n{built.stderr}"
    )
    assert not poison_log.exists(), (
        "the failing shim must never be consulted; the exported prefix has "
        f"to come first on PATH:\n{poison_log.read_text(encoding='utf-8')}"
    )


def test_removing_the_path_export_stops_the_installed_linker_being_found(
    tmp_path: Path,
) -> None:
    """The PATH export must be what selects the default-prefix ``ld.mold``.

    This is the mutation the first contract cannot perform on itself: with
    the export removed and a failing ``ld.mold`` first on ``PATH``, a build
    that still succeeded would prove the export was decorative. The shim
    records its calls, so the assertion reads what the linker consulted
    rather than inferring it from an exit status.
    """
    if not _mold_installable():
        pytest.skip("the linker route needs gcc, which this host does not provide")
    _skip_unless_pin_installed("the linker route links with the pinned nightly")
    installer = prepare(tmp_path, default_prefix=True, request=_linux_request())
    installed = installer.install()
    assert installed.returncode == 0, installed.stderr

    directory = write_probe_repository(tmp_path / "probe")
    (directory / "Makefile").write_text(
        _without_path_export(MAKEFILE_PATH.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    poison_log = tmp_path / "poison-log"
    environment = probe_environment(directory, installer.environment.home)
    environment["PATH"] = (
        f"{_poisoned_path(tmp_path / 'poison', poison_log)}"
        f"{os.pathsep}{installer.environment.values['PATH']}"
    )

    built = run_development_target(directory, environment, "dev-build")

    assert poison_log.is_file() and poison_log.read_text(encoding="utf-8").strip(), (
        "the failing ld.mold must be consulted, or the case proves nothing"
    )
    assert built.returncode != 0, (
        "without the PATH export the failing ld.mold must stop the build; a "
        f"success means the export is not load-bearing:\n{built.stdout}\n"
        f"{built.stderr}"
    )


def _without_path_export(source: str) -> str:
    """Return the Makefile with its ``PATH`` export removed exactly once."""
    assert source.count(PATH_EXPORT) == 1, (
        f"the PATH export must appear once, found {source.count(PATH_EXPORT)}"
    )
    return source.replace(PATH_EXPORT, "")


def _linux_request() -> Request:
    """Return the Linux/x86_64 request the linker contracts install for."""
    return Request(platform=Platform("Linux", "x86_64"))
