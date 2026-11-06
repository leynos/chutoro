"""Build a private mold installer and the archive it is served.

``scripts/install-mold.sh`` is the one script in this layer that reaches
outside the checkout: it fetches a release archive and unpacks it into the
user's prefix. Each contract therefore needs the same substantial fixture - a
repository-shaped copy, a private prefix, a locally built tarball, and
recorded stand-ins for ``rustup``, ``uname`` and ``curl``. Building it here
keeps ``install_mold_test.py`` to the assertions themselves.

The archive's two independent failure modes live in ``Archive``: a truncated
member list and a stale reported version. Both match their checksum, so the
digest cannot be the only gate.

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
MOLD_VERSION = "2.41.0"
SUPPORTED_MACHINE = "x86_64"
ARCHIVE_NAME = f"mold-{MOLD_VERSION}-{SUPPORTED_MACHINE}-linux.tar.gz"
ARCHIVE_MEMBERS = (
    "bin/mold",
    "bin/ld.mold",
    "lib/mold/mold-wrapper.so",
    "libexec/mold/ld",
)
INSTALLER_FILES = ("scripts/install-mold.sh", "scripts/lib/pinned-download.sh")
MAKE_INPUTS = ("Makefile", "tools/dev-fast/TOOLCHAIN")


@dataclass(frozen=True)
class Platform:
    """One host's ``uname`` answers, which is all the installer branches on."""

    system: str
    machine: str


#: The members every real release archive carries. An ``Archive`` may omit one
#: to model a truncated download that still matched its checksum.
REQUIRED_MEMBERS: tuple[str, ...] = ARCHIVE_MEMBERS


@dataclass(frozen=True)
class Archive:
    """What a release-shaped tarball contains and what its binaries report.

    ``members`` and the two reported versions are the ways a well-checksummed
    archive can still be unfit: a truncated one is missing a file the linker
    route needs, and a stale one reports the wrong version. Both must be
    rejected rather than installed and then reused on the next run.

    The versions are separate fields because the installer probes ``mold`` and
    ``ld.mold`` independently. A single shared version could not show that
    removing either probe leaves the other one as the only gate, so a case
    that staled both binaries at once would pass even with one probe deleted.
    """

    members: tuple[str, ...] = REQUIRED_MEMBERS
    mold_version: str = MOLD_VERSION
    ld_mold_version: str = MOLD_VERSION


@dataclass(frozen=True)
class Request:
    """One way to ask for an install: the host, and the archive it is served.

    These travel together because they describe a single scenario. Grouping
    them keeps the fixture's signature small and, more usefully, stops a case
    from varying the archive without also stating which host would fetch it.
    """

    platform: Platform = Platform("Linux", SUPPORTED_MACHINE)
    shape: Archive = Archive()
    pin: str | None = None


@dataclass(frozen=True)
class InstallerRun:
    """A private installer, its prefix, and the record of what it downloaded."""

    prefix: Path
    download_log: Path
    rustup_log: Path
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
    to honour the destination. Recording the call is what lets the reuse test
    prove the network was not consulted a second time.
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


def _recording_rustup(home: Path, log: Path) -> None:
    """Install a ``rustup`` that records the toolchain request.

    ``install-dev-fast`` is the only route that provisions the development
    toolchain, so the record proves the target does it and nothing else does.
    """
    _fake_tool(
        home / ".cargo" / "bin",
        "rustup",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> {str(log)!r}
""",
    )


def _mold_archive(
    path: Path,
    archive_arch: str = SUPPORTED_MACHINE,
    *,
    shape: Archive = Archive(),
) -> None:
    """Build a release-shaped archive whose binaries report the pin.

    The installer re-runs both binaries after copying them, so the members are
    scripts that answer ``--version`` the way the real release does. The
    archive's own name carries ``archive_arch`` because the pin is looked up by
    that name: this is how a release for another architecture is modelled.
    """
    def reporting(version: str) -> bytes:
        """Build a stand-in binary that reports `version` from `--version`."""
        return f"#!/usr/bin/env bash\nprintf 'mold {version}\\n'\n".encode()

    contents = {
        "bin/mold": reporting(shape.mold_version),
        "bin/ld.mold": reporting(shape.ld_mold_version),
        "lib/mold/mold-wrapper.so": b"wrapper",
        "libexec/mold/ld": b"#!/usr/bin/env bash\nexit 0\n",
    }
    prefix = f"mold-{MOLD_VERSION}-{archive_arch}-linux"
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
    (root / "scripts" / "install-mold.sh").chmod(0o755)
    (root / "tools" / "mold").mkdir(parents=True, exist_ok=True)
    (root / "tools" / "mold" / "VERSION").write_text(
        f"{MOLD_VERSION}\n", encoding="utf-8"
    )
    (root / "tools" / "mold" / "SHA256SUMS").write_text(
        f"{archive_sha}  {archive_name}\n", encoding="utf-8"
    )
    return root


def prepare(
    tmp_path: Path,
    *,
    request: Request = Request(),
    through_make: bool = False,
) -> InstallerRun:
    """Build a private installer pinned to a locally built archive.

    ``request.pin`` overrides the recorded digest; passing a wrong one models
    an attacker substituting the archive at the pinned URL. ``request.shape``
    alters what the archive contains while the checksum still matches, which
    is the only way to reach the installer's post-download validation.
    ``through_make`` drives the ``install-dev-fast`` recipe instead of the
    script alone.
    """
    home = tmp_path / "home"
    home.mkdir()
    platform = request.platform
    archive_name = f"mold-{MOLD_VERSION}-{platform.machine}-linux.tar.gz"
    archive = tmp_path / archive_name
    _mold_archive(archive, platform.machine, shape=request.shape)
    digest = request.pin or hashlib.sha256(archive.read_bytes()).hexdigest()
    root = _private_repository(tmp_path, digest, archive_name)
    if through_make:
        _copy_inputs(root, MAKE_INPUTS)

    bin_dir = tmp_path / "bin"
    download_log = tmp_path / "downloads"
    rustup_log = tmp_path / "rustup-calls"
    _recording_curl(bin_dir, archive, download_log)
    _recording_uname(bin_dir, system=platform.system, machine=platform.machine)
    _recording_rustup(home, rustup_log)

    prefix = tmp_path / "prefix"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
            "HOME": str(home),
            "MOLD_INSTALL_PREFIX": str(prefix),
        }
    )

    # The private tree holds only the inputs this route reads. Every other
    # parse-time file the Makefile validates is pointed back at the checkout,
    # so an unrelated pin cannot fail the test for the wrong reason.
    argv = (
        [
            "make",
            "--no-print-directory",
            "--file",
            str(root / "Makefile"),
            f"KANI_VERSION_FILE={REPO_ROOT / 'tools' / 'kani' / 'VERSION'}",
            "install-dev-fast",
        ]
        if through_make
        else ["bash", str(root / "scripts" / "install-mold.sh")]
    )

    def run() -> subprocess.CompletedProcess[str]:
        """Run the installer once, capturing everything it writes."""
        return subprocess.run(
            argv,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    return InstallerRun(prefix, download_log, rustup_log, run)
