"""Build a private cargo-deny installer and the archive it is served.

``scripts/install-cargo-deny.sh`` reaches outside the checkout: it fetches a
release archive and unpacks one binary into a user prefix. Each contract needs
the same fixture - a repository-shaped copy, a private home, a locally built
tarball, and recorded stand-ins for ``curl`` and ``uname``. Building it here
keeps ``install_cargo_deny_test.py`` to the assertions themselves.

Unlike the mold fixture, the archive is not reused across contracts: this
installer deliberately re-downloads every time, so an installation left in the
prefix is evidence the checksum gate ran, not a cache to reuse.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO_DENY_VERSION = "0.22.0"
SUPPORTED_MACHINES = ("x86_64", "aarch64")
INSTALLER_FILES = (
    "scripts/install-cargo-deny.sh",
    "scripts/lib/pinned-download.sh",
)


def archive_stem(machine: str) -> str:
    """Return the release stem for one host architecture.

    The installer derives this name from ``uname -m`` and looks its digest up
    in ``SHA256SUMS``, so the fixture must build the same name before it can
    stand in for the release.
    """
    return f"cargo-deny-{CARGO_DENY_VERSION}-{machine}-unknown-linux-musl"


@dataclass(frozen=True)
class Platform:
    """One host's ``uname`` answers, which is all the installer branches on."""

    system: str = "Linux"
    machine: str = "x86_64"


@dataclass(frozen=True)
class Archive:
    """What a release-shaped tarball contains and what its binary reports.

    ``members`` is relative to the versioned release directory. Omitting the
    ``cargo-deny`` member models a truncated download that still matched its
    checksum; a stale reported version models an archive built from the wrong
    tag. The digest cannot be the only gate for either.
    """

    members: tuple[str, ...] = ("cargo-deny",)
    reported_version: str = CARGO_DENY_VERSION


@dataclass(frozen=True)
class Request:
    """One way to ask for an install: the host, and the archive it is served."""

    platform: Platform = Platform()
    shape: Archive = Archive()
    pin: str | None = None


@dataclass(frozen=True)
class InstallerRun:
    """A private installer, its prefix, and the record of what it downloaded."""

    install_dir: Path
    home: Path
    download_log: Path
    run: Callable[[], subprocess.CompletedProcess[str]]

    def install(self) -> subprocess.CompletedProcess[str]:
        """Run the private installer exactly as the Make target would."""
        return self.run()


def _fake_tool(directory: Path, name: str, body: str) -> Path:
    """Write an executable stand-in and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / name
    tool.write_text(body, encoding="utf-8")
    tool.chmod(0o755)
    return tool


def _recording_curl(bin_dir: Path, archive: Path, log: Path) -> None:
    """Install a ``curl`` that records the call and serves the local archive.

    The installer passes ``-sSfL URL -o DESTINATION``, so the stand-in only has
    to honour the destination. The recorded call is what lets the contracts
    prove that a rejected host never downloaded anything, and that a current
    binary did not make the installer skip the verified download.
    """
    _fake_tool(
        bin_dir,
        "curl",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> {str(log)!r}
cp -- {str(archive)!r} "${{@: -1}}"
""",
    )


def _recording_uname(bin_dir: Path, *, system: str, machine: str) -> None:
    """Install a ``uname`` that reports the requested system and machine."""
    _fake_tool(
        bin_dir,
        "uname",
        f"""#!/usr/bin/env bash
set -euo pipefail
case "${{1:-}}" in
  -s) printf '%s\\n' {system!r} ;;
  -m) printf '%s\\n' {machine!r} ;;
  *) printf '%s\\n' {system!r} ;;
esac
""",
    )


def _reporting_binary(version: str) -> bytes:
    """Build a stand-in binary that reports `version` from ``--version``."""
    return f"#!/usr/bin/env bash\nprintf 'cargo-deny {version}\\n'\n".encode()


def _cargo_deny_archive(path: Path, machine: str, *, shape: Archive) -> None:
    """Build a release-shaped archive whose single binary reports the pin.

    The installer probes the staged binary, so the member is a script that
    answers ``--version`` the way the real release does. The archive's own name
    carries ``machine`` because the digest is looked up by that name: this is
    how a release for another architecture is modelled.
    """
    contents = {
        "cargo-deny": _reporting_binary(shape.reported_version),
        "README.md": b"# cargo-deny\n",
        "LICENSE-MIT": b"MIT\n",
        "LICENSE-APACHE": b"Apache-2.0\n",
    }
    prefix = archive_stem(machine)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in shape.members:
            payload = contents[relative]
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = len(payload)
            info.mode = 0o755
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    path.write_bytes(buffer.getvalue())


def _copy_inputs(root: Path, relatives: tuple[str, ...]) -> None:
    """Copy repository inputs into the private tree, creating parents."""
    for relative in relatives:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())


def _private_repository(tmp_path: Path, archive_sha: str, archive_name: str) -> Path:
    """Copy the installer and its pins into a repository-shaped directory.

    The installer resolves its root from its own location, so a private copy is
    what makes the pin, the prefix, and the downloaded archive all test-owned.
    ``archive_name`` is the name the pin manifest must carry, because that is
    the name the installer derives from the host it is told about.
    """
    root = tmp_path / "repo"
    _copy_inputs(root, INSTALLER_FILES)
    (root / "scripts" / "install-cargo-deny.sh").chmod(0o755)
    (root / "tools" / "cargo-deny").mkdir(parents=True, exist_ok=True)
    (root / "tools" / "cargo-deny" / "VERSION").write_text(
        f"{CARGO_DENY_VERSION}\n", encoding="utf-8"
    )
    (root / "tools" / "cargo-deny" / "SHA256SUMS").write_text(
        f"{archive_sha}  {archive_name}\n", encoding="utf-8"
    )
    return root


def prepare(
    tmp_path: Path,
    *,
    request: Request = Request(),
    default_dir: bool = False,
) -> InstallerRun:
    """Build a private installer pinned to a locally built archive.

    ``request.pin`` overrides the recorded digest; passing a wrong one models
    an attacker substituting the archive at the pinned URL. ``request.shape``
    alters what the archive contains while the checksum still matches, which
    is the only way to reach the installer's post-download validation.

    ``default_dir`` leaves ``CARGO_DENY_INSTALL_DIR`` unset, so the installer
    resolves ``${HOME}/.cargo/bin`` exactly as it does on a developer's
    machine. Every other contract overrides the directory, which is what makes
    this the only mode that exercises the default.
    """
    home = tmp_path / "home"
    home.mkdir()
    platform = request.platform
    archive_name = f"{archive_stem(platform.machine)}.tar.gz"
    archive = tmp_path / archive_name
    _cargo_deny_archive(archive, platform.machine, shape=request.shape)
    digest = request.pin or hashlib.sha256(archive.read_bytes()).hexdigest()
    root = _private_repository(tmp_path, digest, archive_name)

    bin_dir = tmp_path / "bin"
    download_log = tmp_path / "downloads"
    _recording_curl(bin_dir, archive, download_log)
    _recording_uname(bin_dir, system=platform.system, machine=platform.machine)

    install_dir = home / ".cargo" / "bin" if default_dir else tmp_path / "prefix"
    values = os.environ.copy()
    # An inherited override would silently redirect every contract below.
    values.pop("CARGO_DENY_INSTALL_DIR", None)
    values.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{values['PATH']}",
            "HOME": str(home),
        }
    )
    if not default_dir:
        values["CARGO_DENY_INSTALL_DIR"] = str(install_dir)

    def run() -> subprocess.CompletedProcess[str]:
        """Run the installer once, capturing everything it writes."""
        return subprocess.run(
            ["bash", str(root / "scripts" / "install-cargo-deny.sh")],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=values,
        )

    return InstallerRun(install_dir, home, download_log, run)