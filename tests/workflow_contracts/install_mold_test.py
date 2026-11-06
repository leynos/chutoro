"""Integration contracts for the pinned mold installer and its Make target.

``make install-dev-fast`` is the one command in this layer that reaches
outside the checkout: it installs a toolchain and unpacks a release archive
into the user's prefix. These tests run the real ``scripts/install-mold.sh``
and the real ``install-dev-fast`` recipe against a private repository copy, a
private install prefix, and recorded stand-ins for ``rustup``, ``uname`` and
``curl``, so every branch is exercised without a network call and without
touching ``~/.local``.

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

import pytest

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

    ``members`` and ``reported_version`` are the two ways a well-checksummed
    archive can still be unfit: a truncated one is missing a file the linker
    route needs, and a stale one reports the wrong version. Both must be
    rejected rather than installed and then reused on the next run.
    """

    members: tuple[str, ...] = REQUIRED_MEMBERS
    reported_version: str = MOLD_VERSION


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
    script = (
        f"#!/usr/bin/env bash\nprintf 'mold {shape.reported_version}\\n'\n"
    ).encode()
    contents = {
        "bin/mold": script,
        "bin/ld.mold": script,
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


def _prepare(
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
        return subprocess.run(
            argv,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    return InstallerRun(prefix, download_log, rustup_log, run)


def test_installer_unpacks_the_verified_archive_into_the_prefix(
    tmp_path: Path,
) -> None:
    """A cold install must place every file the linker route needs."""
    installer = _prepare(tmp_path)

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    assert f"Installed mold {MOLD_VERSION} at" in completed.stdout
    for relative in ARCHIVE_MEMBERS:
        assert (installer.prefix / relative).exists(), (
            f"installer did not place {relative}"
        )


def test_aarch64_host_selects_the_aarch64_archive(tmp_path: Path) -> None:
    """The architecture branch must reach the matching pinned release.

    The pin manifest is looked up by the archive name the installer derives
    from ``uname -m``. A host reporting ``aarch64`` that requested the x86_64
    name would find no pin and stop, and one that requested it and found a pin
    would install the wrong binary; the recorded download proves which name
    was actually fetched.
    """
    installer = _prepare(
        tmp_path, request=Request(platform=Platform("Linux", "aarch64"))
    )

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    requested = installer.download_log.read_text(encoding="utf-8")
    assert "mold-2.41.0-aarch64-linux.tar.gz" in requested, (
        f"an aarch64 host must request the aarch64 archive: {requested!r}"
    )
    assert "x86_64" not in requested, (
        f"an aarch64 host must not fetch the x86_64 archive: {requested!r}"
    )
    assert (installer.prefix / "bin" / "mold").exists()


def test_complete_installation_is_reused_without_downloading(tmp_path: Path) -> None:
    """A warm image must not pay for the download again."""
    installer = _prepare(tmp_path)
    assert installer.install().returncode == 0
    installer.download_log.unlink()

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    assert f"mold {MOLD_VERSION} already installed at" in completed.stdout
    assert not installer.download_log.exists(), (
        "a complete installation must not download again"
    )
    assert (installer.prefix / "bin" / "ld.mold").exists()


def test_incomplete_installation_is_replaced(tmp_path: Path) -> None:
    """A half-populated prefix must fall through and be repaired.

    An image that restored only part of the prefix used to look installed; the
    reuse check exists so that case reinstalls instead of failing later at link
    time.
    """
    installer = _prepare(tmp_path)
    (installer.prefix / "bin").mkdir(parents=True)
    for relative in ("bin/mold", "bin/ld.mold"):
        stub = installer.prefix / relative
        stub.write_text(
            "#!/usr/bin/env bash\nprintf 'mold 2.41.0\\n'\n", encoding="utf-8"
        )
        stub.chmod(0o755)

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    assert installer.download_log.exists(), (
        "an incomplete installation must be replaced"
    )
    assert (installer.prefix / "lib" / "mold" / "mold-wrapper.so").exists()
    assert (installer.prefix / "libexec" / "mold" / "ld").exists()


def test_checksum_mismatch_fails_without_installing(tmp_path: Path) -> None:
    """The pin is load-bearing: a substituted archive must not be unpacked."""
    installer = _prepare(tmp_path, request=Request(pin="0" * 64))

    completed = installer.install()

    assert completed.returncode != 0
    assert "SHA-256 mismatch" in completed.stderr
    assert not (installer.prefix / "bin").exists()


@pytest.mark.parametrize(
    ("shape", "expected"),
    (
        pytest.param(
            Archive(members=tuple(m for m in REQUIRED_MEMBERS if m != "bin/ld.mold")),
            "does not contain an executable bin/ld.mold",
            id="missing-ld-mold",
        ),
        pytest.param(
            Archive(members=("bin/mold", "bin/ld.mold", "libexec/mold/ld")),
            "missing lib/mold/mold-wrapper.so",
            id="missing-wrapper",
        ),
        pytest.param(
            Archive(members=("bin/mold", "bin/ld.mold", "lib/mold/mold-wrapper.so")),
            "missing libexec/mold/ld",
            id="missing-libexec",
        ),
        pytest.param(
            Archive(reported_version="2.40.0"),
            "Staged mold does not report 2.41.0",
            id="wrong-version",
        ),
    ),
)
def test_unfit_archive_with_a_matching_checksum_is_rejected(
    tmp_path: Path, shape: Archive, expected: str
) -> None:
    """A correct digest does not make an archive installable.

    A truncated or stale archive can still match the pin, so the checksum
    cannot be the only gate. Each case must fail with its own diagnostic and
    leave no installed tree, because a partial prefix that looked installed
    would be reused by the next run and fail later at link time.
    """
    installer = _prepare(tmp_path, request=Request(shape=shape))

    completed = installer.install()

    assert completed.returncode != 0, (
        "an unfit archive must not be installed: "
        f"{completed.stdout}\n{completed.stderr}"
    )
    assert expected in completed.stderr, (
        f"expected {expected!r}, got: {completed.stderr!r}"
    )
    assert not (installer.prefix / "bin").exists(), (
        "a rejected archive must not leave an installed binary behind"
    )


@pytest.mark.parametrize(
    ("platform", "expected"),
    (
        pytest.param(
            Platform("Darwin", "arm64"),
            "Unsupported mold operating system",
            id="mac",
        ),
        pytest.param(
            Platform("Linux", "riscv64"),
            "Unsupported Linux architecture",
            id="arch",
        ),
    ),
)
def test_unsupported_platform_is_a_hard_error(
    tmp_path: Path, platform: Platform, expected: str
) -> None:
    """Every unsupported platform must stop rather than install something wrong."""
    installer = _prepare(tmp_path, request=Request(platform=platform))

    completed = installer.install()

    assert completed.returncode != 0
    assert expected in completed.stderr
    assert not installer.download_log.exists(), (
        "an unsupported platform must be rejected before downloading"
    )


def test_make_install_dev_fast_installs_the_toolchain_and_mold(
    tmp_path: Path,
) -> None:
    """The Make target must provision both halves in one command.

    Callers only ever run ``make install-dev-fast``, so the recipe - not the
    installer script - is the contract: it must install the pinned toolchain
    components and then hand off to the verified download.
    """
    installer = _prepare(tmp_path, through_make=True)

    completed = installer.install()

    assert completed.returncode == 0, (
        f"install-dev-fast failed: {completed.stdout}\n{completed.stderr}"
    )
    rustup_calls = installer.rustup_log.read_text(encoding="utf-8").splitlines()
    assert len(rustup_calls) == 1, f"expected one rustup call: {rustup_calls!r}"
    assert rustup_calls[0].startswith("toolchain install nightly-"), (
        f"the target must install the pinned development toolchain: {rustup_calls!r}"
    )
    for component in (
        "rustfmt",
        "clippy",
        "rust-analyzer",
        "rustc-codegen-cranelift",
    ):
        assert f"--component {component}" in rustup_calls[0], (
            f"the toolchain install lost {component!r}: {rustup_calls[0]!r}"
        )
    assert (installer.prefix / "bin" / "mold").exists()


def test_make_install_dev_fast_leaves_mold_alone_off_linux(tmp_path: Path) -> None:
    """A non-Linux host must still get the toolchain, but never a mold install.

    The recipe branches on the host OS, so an unguarded hand-off would fail on
    every macOS or Windows contributor in a way the Rust gates never would.
    """
    installer = _prepare(
        tmp_path,
        request=Request(platform=Platform("Darwin", "arm64")),
        through_make=True,
    )

    completed = installer.install()

    assert completed.returncode == 0, (
        f"install-dev-fast failed: {completed.stdout}\n{completed.stderr}"
    )
    assert "mold is Linux-only" in completed.stdout
    assert installer.rustup_log.exists(), "the toolchain install must still run"
    assert not installer.download_log.exists(), "mold must not be downloaded"
    assert not installer.prefix.exists(), "nothing may be installed off Linux"
