"""Integration contracts for the pinned mold installer and its Make target.

``make install-dev-fast`` is the one command in this layer that reaches
outside the checkout: it installs a toolchain and unpacks a release archive
into the user's prefix. Each contract here asserts one way that install can
go wrong, or one half of the route that must keep working.

The private repository, prefix, archive, and recording stand-ins those
contracts share are built by ``mold_install_support``, so what remains is the
assertions.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mold_install_support import (
    ARCHIVE_MEMBERS,
    MOLD_VERSION,
    REQUIRED_MEMBERS,
    Archive,
    Platform,
    Request,
    prepare,
)


def test_installer_unpacks_the_verified_archive_into_the_prefix(
    tmp_path: Path,
) -> None:
    """A cold install must place every file the linker route needs."""
    installer = prepare(tmp_path)

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
    installer = prepare(
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
    installer = prepare(tmp_path)
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
    installer = prepare(tmp_path)
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
    installer = prepare(tmp_path, request=Request(pin="0" * 64))

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
    installer = prepare(tmp_path, request=Request(shape=shape))

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
    installer = prepare(tmp_path, request=Request(platform=platform))

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
    installer = prepare(tmp_path, through_make=True)

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
    installer = prepare(
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
