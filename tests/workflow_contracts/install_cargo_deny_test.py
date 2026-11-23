"""Integration contracts for the pinned cargo-deny installer.

The installer fetches a release archive and unpacks one binary into a prefix.
Each contract here asserts one way that install can go wrong, or one half of
the route that must keep working: cold installation, either supported Linux
architecture, the default and overridden directories, and the digest, member
and version gates that make the archive trustworthy.

The private repository, prefix, archive, and recording stand-ins those
contracts share are built by ``cargo_deny_install_support``, so what remains
is the assertions.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from cargo_deny_install_support import (
    CARGO_DENY_VERSION,
    SUPPORTED_MACHINES,
    Archive,
    InstallerRun,
    Platform,
    Request,
    archive_stem,
    prepare,
)


def _installed_binary(installer: InstallerRun) -> Path:
    """Return the binary path inside an installer run's prefix."""
    return installer.install_dir / "cargo-deny"


def _assert_nothing_installed(install_dir: Path) -> None:
    """Require a rejected archive to leave the prefix absent or empty."""
    if install_dir.exists():
        assert not any(install_dir.iterdir()), (
            f"a rejected archive left an installed tree: "
            f"{sorted(install_dir.rglob('*'))}"
        )


def test_cold_installation_places_the_pinned_binary(tmp_path: Path) -> None:
    """A cold install must place an executable that reports the pinned version."""
    installer = prepare(tmp_path)

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    assert f"Installed cargo-deny {CARGO_DENY_VERSION} at" in completed.stdout
    binary = _installed_binary(installer)
    assert binary.is_file(), "the installer did not place cargo-deny"
    reported = subprocess.run(
        [str(binary), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reported.stdout.strip() == f"cargo-deny {CARGO_DENY_VERSION}"


@pytest.mark.parametrize("machine", SUPPORTED_MACHINES)
def test_each_supported_linux_host_requests_its_own_archive(
    tmp_path: Path, machine: str
) -> None:
    """The architecture branch must reach the matching pinned release.

    The pin manifest is looked up by the archive name the installer derives
    from ``uname -m``. A host reporting ``aarch64`` that requested the x86_64
    name would find no pin and stop; the recorded download proves which name
    was actually fetched.
    """
    installer = prepare(tmp_path, request=Request(platform=Platform(machine=machine)))

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    requested = installer.download_log.read_text(encoding="utf-8")
    assert f"{archive_stem(machine)}.tar.gz" in requested, (
        f"a {machine} host must request its own archive: {requested!r}"
    )
    other = next(name for name in SUPPORTED_MACHINES if name != machine)
    assert other not in requested, (
        f"a {machine} host must not fetch the {other} archive: {requested!r}"
    )
    assert _installed_binary(installer).exists()


def test_install_directory_can_be_overridden(tmp_path: Path) -> None:
    """The documented override must decide where the binary lands."""
    installer = prepare(tmp_path)

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    assert installer.install_dir == tmp_path / "prefix"
    assert _installed_binary(installer).is_file()


def test_default_install_directory_is_the_cargo_bin_directory(
    tmp_path: Path,
) -> None:
    """Without an override the binary must be where `cargo deny` searches.

    The Makefile puts ``${HOME}/.cargo/bin`` ahead of the rest of ``PATH``, so
    an installer default anywhere else would leave `make deny` resolving some
    other cargo-deny.
    """
    installer = prepare(tmp_path, default_dir=True)

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    assert installer.install_dir == installer.home / ".cargo" / "bin"
    assert _installed_binary(installer).is_file()


def test_checksum_mismatch_fails_without_installing(tmp_path: Path) -> None:
    """The pin is load-bearing: a substituted archive must not be unpacked."""
    installer = prepare(tmp_path, request=Request(pin="0" * 64))

    completed = installer.install()

    assert completed.returncode != 0
    assert "SHA-256 mismatch" in completed.stderr
    _assert_nothing_installed(installer.install_dir)


def test_archive_without_the_binary_is_rejected(tmp_path: Path) -> None:
    """A digest-valid archive that lacks the binary must not install.

    A truncated release can still match its checksum, so the member list is a
    second gate rather than a formality.
    """
    installer = prepare(
        tmp_path,
        request=Request(shape=Archive(members=("README.md",))),
    )

    completed = installer.install()

    assert completed.returncode != 0
    assert "cargo-deny archive does not contain cargo-deny" in completed.stderr
    _assert_nothing_installed(installer.install_dir)


def test_wrong_staged_version_is_rejected(tmp_path: Path) -> None:
    """A digest-valid archive built from the wrong tag must not install.

    Checking only that the archive contains a binary would let a mislabelled
    release through; the staged probe is what ties the archive to the pin.
    """
    installer = prepare(
        tmp_path,
        request=Request(shape=Archive(reported_version="0.21.0")),
    )

    completed = installer.install()

    assert completed.returncode != 0
    assert f"Staged cargo-deny does not report {CARGO_DENY_VERSION}" in (
        completed.stderr
    )
    _assert_nothing_installed(installer.install_dir)


@pytest.mark.parametrize(
    ("platform", "expected"),
    (
        pytest.param(
            Platform("Darwin", "arm64"),
            "Unsupported cargo-deny operating system",
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
    """Every unsupported platform must stop before touching the network."""
    installer = prepare(tmp_path, request=Request(platform=platform))

    completed = installer.install()

    assert completed.returncode != 0
    assert expected in completed.stderr
    assert not installer.download_log.exists(), (
        "an unsupported platform must be rejected before downloading"
    )
    _assert_nothing_installed(installer.install_dir)


def test_a_current_binary_is_still_replaced_from_the_verified_archive(
    tmp_path: Path,
) -> None:
    """A matching version string must not be accepted as proof of provenance.

    A version string costs an attacker nothing, so this installer always
    fetches and verifies the archive. The seeded binary reports the pin but
    leaves a sentinel when run; the installed file must be the archive's.
    """
    installer = prepare(tmp_path)
    installer.install_dir.mkdir(parents=True)
    sentinel = tmp_path / "seeded-binary-ran"
    seeded = installer.install_dir / "cargo-deny"
    seeded.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'cargo-deny {CARGO_DENY_VERSION}\\n'\n"
        f"touch {str(sentinel)!r}\n",
        encoding="utf-8",
    )
    seeded.chmod(0o755)

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    assert installer.download_log.exists(), (
        "a current version string must not skip the verified download"
    )
    assert "already installed" not in completed.stdout
    assert "seeded-binary-ran" not in seeded.read_text(encoding="utf-8"), (
        "the verified archive must replace the seeded binary"
    )
    assert not sentinel.exists(), "the seeded binary must never run"