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

import subprocess
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


WARM_REUSE_SUPPORT_FILES = (
    "lib/mold/mold-wrapper.so",
    "libexec/mold/ld",
)


def _write_warm_binary(path: Path, version: str) -> None:
    """Place an executable that reports ``version`` at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!/usr/bin/env bash\nprintf 'mold {version}\\n'\n", encoding="utf-8"
    )
    path.chmod(0o755)


def _seed_warm_installation(
    prefix: Path, *, mold_version: str, ld_mold_version: str
) -> None:
    """Seed a complete-looking prefix whose two binaries report what is asked.

    Both support files are present, so each case isolates one binary's version
    probe rather than overlapping with the missing-member cases.
    """
    _write_warm_binary(prefix / "bin" / "mold", mold_version)
    _write_warm_binary(prefix / "bin" / "ld.mold", ld_mold_version)
    for relative in WARM_REUSE_SUPPORT_FILES:
        support = prefix / relative
        support.parent.mkdir(parents=True, exist_ok=True)
        support.write_text("stale\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("mold_version", "ld_mold_version", "repaired"),
    (
        pytest.param(MOLD_VERSION, MOLD_VERSION, False, id="both-current"),
        pytest.param("2.40.0", MOLD_VERSION, True, id="stale-mold"),
        pytest.param(MOLD_VERSION, "2.40.0", True, id="stale-ld-mold"),
        pytest.param("2.40.0", "2.40.0", True, id="stale-both"),
    ),
)
def test_warm_installation_is_reused_only_when_both_binaries_are_current(
    tmp_path: Path, mold_version: str, ld_mold_version: str, repaired: bool
) -> None:
    """Each warm-reuse probe must be load-bearing on its own.

    The linker route runs ``ld.mold`` as well as ``mold``, so a prefix can hold
    one current binary beside one stale one. Testing only the stale-both case
    would pass with either probe deleted, so each binary is staled alone.
    """
    installer = prepare(tmp_path)
    _seed_warm_installation(
        installer.prefix,
        mold_version=mold_version,
        ld_mold_version=ld_mold_version,
    )

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    if repaired:
        assert installer.download_log.exists(), (
            f"a prefix reporting mold={mold_version!r} and "
            f"ld.mold={ld_mold_version!r} must be replaced"
        )
        # A download alone does not prove the repair succeeded. The reuse check
        # exists to keep a stale pair out of the linker route, so the contract
        # that a stale prefix is repaired must also assert the *result*: both
        # binaries report the pinned version afterwards. Without this, deleting
        # the post-copy probe would still satisfy every assertion above.
        for binary in ("mold", "ld.mold"):
            installed = installer.prefix / "bin" / binary
            assert installed.is_file(), (
                f"{binary} is missing after repairing the stale prefix"
            )
            reported = subprocess.run(
                [str(installed), "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
            assert reported.returncode == 0, (
                f"{binary} --version failed after the repair: {reported.stderr!r}"
            )
            assert reported.stdout.startswith(f"mold {MOLD_VERSION}"), (
                f"{binary} reports {reported.stdout.strip()!r} after the repair, "
                f"not the pinned mold {MOLD_VERSION}"
            )
    else:
        assert not installer.download_log.exists(), (
            "a complete installation must not download again"
        )
        assert "already installed" in completed.stdout


@pytest.mark.parametrize("missing", ARCHIVE_MEMBERS)
def test_incomplete_installation_is_replaced(
    tmp_path: Path, missing: str
) -> None:
    """A half-populated prefix must fall through and be repaired.

    An image that restored only part of the prefix used to look installed; the
    reuse check exists so that case reinstalls instead of failing later at link
    time. Each member is omitted alone so every part of the reuse check is
    exercised, not just the first one that fails.
    """
    installer = prepare(tmp_path)
    (installer.prefix / "bin").mkdir(parents=True)
    for relative in ARCHIVE_MEMBERS:
        if relative == missing:
            continue
        _write_warm_binary(installer.prefix / relative, MOLD_VERSION)

    completed = installer.install()

    assert completed.returncode == 0, completed.stderr
    assert installer.download_log.exists(), (
        f"a prefix missing {missing} must be replaced"
    )
    for relative in ARCHIVE_MEMBERS:
        assert (installer.prefix / relative).exists(), (
            f"the repair did not restore {relative}"
        )


def test_checksum_mismatch_fails_without_installing(tmp_path: Path) -> None:
    """The pin is load-bearing: a substituted archive must not be unpacked."""
    installer = prepare(tmp_path, request=Request(pin="0" * 64))

    completed = installer.install()

    assert completed.returncode != 0
    assert "SHA-256 mismatch" in completed.stderr
    assert not installer.prefix.exists(), (
        "a rejected archive must not create the install prefix"
    )


def _assert_nothing_installed(prefix: Path) -> None:
    """Require a rejected archive to leave the prefix absent or empty."""
    if prefix.exists():
        assert not any(prefix.iterdir()), (
            f"a rejected archive left an installed tree: {sorted(prefix.rglob('*'))}"
        )


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
        # One binary at a time: the installer probes both, and a case that
        # staled both together would still pass with either probe removed.
        pytest.param(
            Archive(mold_version="2.40.0"),
            "Staged mold does not report 2.41.0",
            id="wrong-mold-version",
        ),
        pytest.param(
            Archive(ld_mold_version="2.40.0"),
            "Staged ld.mold does not report 2.41.0",
            id="wrong-ld-mold-version",
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
    _assert_nothing_installed(installer.prefix)


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
