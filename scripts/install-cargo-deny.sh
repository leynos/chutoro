#!/usr/bin/env bash
#
# Install the pinned cargo-deny release binary for supported Linux hosts.
#
# `make deny` and the `cargo-deny` CI job need the same tool, so the version
# and every supported archive digest are pinned under `tools/cargo-deny/`.
# The upstream release publishes musl builds only, which have no glibc floor
# and therefore run unchanged on GitHub's and Ubicloud's images.
#
# Unlike the other installers in this directory, this one never reuses a
# binary because it reports the pinned version. A version string costs an
# attacker nothing, so a tampered binary would pass that probe; the archive
# digest is the only gate that means anything, and it is always consulted.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/pinned-download.sh
source "${ROOT_DIR}/scripts/lib/pinned-download.sh"

CARGO_DENY_VERSION="$(pinned_version "${ROOT_DIR}/tools/cargo-deny/VERSION")"
CARGO_DENY_OS="$(uname -s)"
[[ "${CARGO_DENY_OS}" == Linux ]] ||
  pinned_die "Unsupported cargo-deny operating system: ${CARGO_DENY_OS} (Linux required)"

CARGO_DENY_HOST_ARCH="$(uname -m)"
case "${CARGO_DENY_HOST_ARCH}" in
  x86_64)
    CARGO_DENY_ARCHIVE_ARCH="x86_64"
    ;;
  aarch64)
    CARGO_DENY_ARCHIVE_ARCH="aarch64"
    ;;
  *)
    pinned_die "Unsupported Linux architecture for cargo-deny: ${CARGO_DENY_HOST_ARCH} (supported: x86_64, aarch64)"
    ;;
esac

# `~/.cargo/bin` is the directory `cargo` searches first for a subcommand, and
# the Makefile puts it ahead of the rest of PATH, so the default install is the
# binary `make deny` resolves. CI overrides this with a job-specific directory
# so the runner image's own cargo-deny cannot shadow the pin.
INSTALL_DIR="${CARGO_DENY_INSTALL_DIR:-${HOME}/.cargo/bin}"
BINARY="${INSTALL_DIR}/cargo-deny"
STEM="cargo-deny-${CARGO_DENY_VERSION}-${CARGO_DENY_ARCHIVE_ARCH}-unknown-linux-musl"
ARCHIVE="${STEM}.tar.gz"
URL="https://github.com/EmbarkStudios/cargo-deny/releases/download/${CARGO_DENY_VERSION}/${ARCHIVE}"

reports_pinned_version() {
  "$1" --version 2>/dev/null |
    grep -qFx "cargo-deny ${CARGO_DENY_VERSION}"
}

EXPECTED_SHA="$(pinned_expected_sha "${ROOT_DIR}/tools/cargo-deny/SHA256SUMS" "${ARCHIVE}")"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT

pinned_fetch_verified "${URL}" "${TMP_DIR}/${ARCHIVE}" "${EXPECTED_SHA}"
tar -tzf "${TMP_DIR}/${ARCHIVE}" >/dev/null ||
  pinned_die "Invalid cargo-deny archive: ${ARCHIVE}"

STAGE_DIR="${TMP_DIR}/cargo-deny"
mkdir -p "${STAGE_DIR}"
tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "${STAGE_DIR}" --strip-components=1
# The staged binary is validated before the install directory is touched, so
# a truncated or stale archive that still matched its checksum cannot leave a
# half-installed tree behind for the next run to reuse.
[[ -f "${STAGE_DIR}/cargo-deny" ]] ||
  pinned_die "cargo-deny archive does not contain cargo-deny"
chmod +x "${STAGE_DIR}/cargo-deny"
reports_pinned_version "${STAGE_DIR}/cargo-deny" ||
  pinned_die "Staged cargo-deny does not report ${CARGO_DENY_VERSION}"

mkdir -p "${INSTALL_DIR}"
mv -f "${STAGE_DIR}/cargo-deny" "${BINARY}"

reports_pinned_version "${BINARY}" ||
  pinned_die "Installed cargo-deny does not report ${CARGO_DENY_VERSION}"
echo "Installed cargo-deny ${CARGO_DENY_VERSION} at ${BINARY}"