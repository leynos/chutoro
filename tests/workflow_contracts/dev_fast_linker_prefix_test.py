"""Contract for the installer's default prefix and the PATH that reaches it.

``scripts/install-mold.sh`` installs to ``${HOME}/.local`` unless
``MOLD_INSTALL_PREFIX`` overrides it, and every other installer contract
overrides it. The Makefile's ``export PATH`` line is what makes that default
prefix usable: ``gcc`` resolves ``ld.mold`` from ``PATH``, so without the line
a developer who ran ``make install-dev-fast`` would still link with whichever
mold the host happened to provide.

Two contracts pin each half. The first installs with the default prefix and
links a real crate, proving the installed tree is one the linker route
accepts. It replaces the fixture's version-reporting ``ld.mold`` with a proxy
that records its call and then runs the host's real linker, so the build
produces a runnable artifact and the contract can read which path ``gcc``
selected rather than infer it from a success. The second is its control: with
the ``PATH`` line removed and a failing ``ld.mold`` ahead of the host's own,
the same build must not reach the installed binary. Without that control the
first contract would pass on a host that already had a working mold anywhere
else.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from dev_fast_make_support import MAKEFILE_PATH, _skip_unless_pin_installed
from dev_fast_probe_support import (
    PROBE_CRATE,
    PROBE_GREETING,
    probe_environment,
    run_development_target,
    write_probe_repository,
)
from mold_install_support import Environment, Platform, Request, prepare

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

#: An ``ld.mold`` that records the path it was invoked through and then hands
#: the link to the host's real linker. The installed archive carries a
#: version-reporting stand-in that links nothing, so a build using it would
#: report success without producing an artifact. Recording before the handoff
#: lets the contract read which ``ld.mold`` the link resolved, and delegating
#: lets it check the artifact that came out.
LINKER_PROXY = """#!/usr/bin/env bash
printf '%s\\n' "$0" >> __LOG__
exec __REAL__ "$@"
"""


def _host_linker() -> Path | None:
    """Return the host's own ``ld.mold``, which the proxy delegates to."""
    found = shutil.which("ld.mold")
    return Path(found) if found is not None else None


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


def _recording_proxy(path: Path, log: Path, real_linker: Path) -> None:
    """Replace an installed ``ld.mold`` with one that records and delegates.

    The installer verifies the binaries it stages, so the proxy is written
    after the install rather than served in the archive. It keeps the
    installer's own prefixes and pins under test, and turns the compile into
    a read of which ``ld.mold`` the recipe actually selected.
    """
    body = (
        LINKER_PROXY.replace("__LOG__", str(log))
        .replace("__REAL__", str(real_linker))
    )
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_the_installer_places_a_default_prefix_tree_the_linker_loads(
    tmp_path: Path,
) -> None:
    """``install-dev-fast`` must install somewhere the recipe can then find.

    The installer's default is ``${HOME}/.local`` and the recipe reaches that
    directory only through its ``PATH`` export. Running both against one
    private home is what joins those two halves: the install proves the
    default prefix is populated, and the build proves the exported ``PATH``
    is what the linker consults.

    Two things stop a bare success from meaning anything. A host may already
    carry a working mold, so a failing shim ahead of everything makes the
    success depend on the export. And the fixture's installed ``ld.mold`` is
    a stand-in that links nothing, so a build reaching it could still report
    success without an artifact; the proxy therefore records the path the
    link resolved and delegates to the host's real linker, and the contract
    reads both the record and the binary it produced.
    """
    if not _mold_installable():
        pytest.skip("the linker route needs gcc, which this host does not provide")
    real_linker = _host_linker()
    if real_linker is None:
        pytest.skip("the linker route needs ld.mold, which this host does not provide")
    _skip_unless_pin_installed(
        "the linker route links with the pinned nightly", needs_cranelift=True
    )
    installer = prepare(tmp_path, default_prefix=True, request=_linux_request())

    installed = installer.install()

    assert installed.returncode == 0, installed.stderr
    installed_linker = installer.prefix / "bin" / "ld.mold"
    assert installed_linker.is_file(), (
        f"the default prefix must hold ld.mold: {installer.prefix}"
    )
    link_log = tmp_path / "link-log"
    _recording_proxy(installed_linker, link_log, real_linker)

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
    selected = link_log.read_text(encoding="utf-8").split()
    assert selected and set(selected) == {str(installed_linker)}, (
        "every link must resolve the developer's installed ld.mold, not the "
        f"host's; expected only {installed_linker!s}, got {selected!r}"
    )
    binary = directory / "target" / "debug" / PROBE_CRATE
    assert binary.is_file(), f"the recipe produced no artifact at {binary}"
    ran = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert ran.stdout.strip() == PROBE_GREETING, (
        "the artifact the installed linker produced must run: "
        f"{ran.returncode} {ran.stdout!r} {ran.stderr!r}"
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
    _skip_unless_pin_installed(
        "the linker route links with the pinned nightly", needs_cranelift=True
    )
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


#: A recipe asking the loader where ``ld.mold`` would come from. It reads the
#: installed tree through ``PATH``, which is the only way the development
#: recipes reach it, and prints the resolved path or ``MISSING``.
RESOLVE_RECIPE = """probe:
\t@command -v ld.mold || echo MISSING
"""

#: The program the probe runs from the clean ``PATH``. Everything else —
#: including any mold the developer's own path carries — stays out of it.
#: ``make`` is the one tool that has to be a real executable, because the test
#: invokes it by name. ``command``, ``echo`` and ``printf`` are shell builtins
#: that ``sh`` resolves itself, so requiring them as binaries would fail on a
#: host that has no such files, which is what a CI runner turned out to be.
PROBE_TOOL = "make"


def _clean_binary_directory(directory: Path) -> Path:
    """Return a ``PATH`` entry holding the probe's tool and no linker.

    An inherited ``PATH`` is what makes a first attempt at this contract prove
    nothing: this host ships ``/usr/bin/ld.mold``, so both halves resolved a
    linker and the export looked decorative. The entry therefore carries only
    ``make``, symlinked by absolute path; the recipe it runs needs nothing
    else that is not a shell builtin.
    """
    directory.mkdir(parents=True, exist_ok=True)
    found = shutil.which(PROBE_TOOL)
    assert found is not None, f"the probe needs {PROBE_TOOL} on this host"
    (directory / PROBE_TOOL).symlink_to(found)
    return directory


def _resolve_under_export(
    directory: Path, environment: Environment
) -> subprocess.CompletedProcess[str]:
    """Ask a Makefile carrying only the export where ``ld.mold`` resolves."""
    makefile = directory / "Makefile"
    # ``PATH_EXPORT`` is the assignment line without its newline, so the
    # separator has to be restored or the recipe joins the same line.
    makefile.write_text(f"{PATH_EXPORT}\n{RESOLVE_RECIPE}", encoding="utf-8")
    values = dict(environment.values)
    values["PATH"] = str(_clean_binary_directory(directory / "clean-bin"))
    return subprocess.run(
        ["make", "--no-print-directory", "--file", str(makefile), "probe"],
        cwd=directory,
        env=values,
        check=False,
        capture_output=True,
        text=True,
    )


def test_the_default_prefix_linker_resolves_only_through_the_path_export(
    tmp_path: Path,
) -> None:
    """The installed linker must be reachable, and only via the exported line.

    The compiling contracts need the pinned nightly and the Cranelift
    component, so on a host without them they skip and this half of the
    install goes unchecked. This contract asks the cheaper question instead:
    with a private home and a ``PATH`` that carries no linker at all, does a
    Makefile holding only the export resolve the installed ``ld.mold``?
    Running the same recipe with the export removed must then fail, or the
    first answer came from the host rather than from the installed tree.
    """
    if not _mold_installable():
        pytest.skip("the installer needs gcc, which this host does not provide")
    installer = prepare(tmp_path, default_prefix=True, request=_linux_request())
    installed = installer.install()
    assert installed.returncode == 0, installed.stderr

    directory = tmp_path / "exported"
    directory.mkdir()
    resolved = _resolve_under_export(directory, installer.environment)

    expected = installer.prefix / "bin" / "ld.mold"
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.strip() == str(expected), (
        "the export must resolve the developer's installed linker, not the "
        f"host's; expected {expected}, got {resolved.stdout.strip()!r}"
    )

    control = tmp_path / "unexported"
    control.mkdir()
    (control / "Makefile").write_text(RESOLVE_RECIPE, encoding="utf-8")
    values = dict(installer.environment.values)
    values["PATH"] = str(_clean_binary_directory(control / "clean-bin"))
    unresolved = subprocess.run(
        ["make", "--no-print-directory", "--file", str(control / "Makefile"), "probe"],
        cwd=control,
        env=values,
        check=False,
        capture_output=True,
        text=True,
    )

    assert unresolved.returncode == 0, unresolved.stderr
    assert unresolved.stdout.strip() == "MISSING", (
        "without the export no linker may resolve from the clean PATH; a hit "
        f"means the first half proved nothing: {unresolved.stdout.strip()!r}"
    )
