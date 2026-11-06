#!/usr/bin/env bash
#
# Install the pinned mold release binary for supported Linux architectures.
#
# CI uses the upstream binary-only archive so installing mold never builds
# it from source. The archive's SHA-256 is pinned in tools/mold/SHA256SUMS;
# unsupported platforms and checksum mismatches are hard errors.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/pinned-download.sh
source "${ROOT_DIR}/scripts/lib/pinned-download.sh"

MOLD_VERSION="$(pinned_version "${ROOT_DIR}/tools/mold/VERSION")"
MOLD_OS="$(uname -s)"
[[ "${MOLD_OS}" == Linux ]] ||
  pinned_die "Unsupported mold operating system: ${MOLD_OS} (Linux required)"

MOLD_HOST_ARCH="$(uname -m)"
case "${MOLD_HOST_ARCH}" in
  x86_64)
    MOLD_ARCHIVE_ARCH="x86_64"
    ;;
  aarch64)
    MOLD_ARCHIVE_ARCH="aarch64"
    ;;
  *)
    pinned_die "Unsupported Linux architecture for mold: ${MOLD_HOST_ARCH} (supported: x86_64, aarch64)"
    ;;
esac

INSTALL_PREFIX="${MOLD_INSTALL_PREFIX:-${HOME}/.local}"
BINARY="${INSTALL_PREFIX}/bin/mold"
ARCHIVE="mold-${MOLD_VERSION}-${MOLD_ARCHIVE_ARCH}-linux.tar.gz"
URL="https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/${ARCHIVE}"

# Both binaries are probed, here and again in staging. The linker route runs
# `ld.mold` as well as `mold`, so a prefix restored from a partial backup could
# hold one current binary and one stale one; checking the second only for
# existence would reuse that pair and fail later at link time.
reports_pinned_version() {
  "$1" --version 2>/dev/null |
    grep -qE "^mold ${MOLD_VERSION//./\\.}([[:space:]]|$)"
}

# A warm image can reuse a complete install. An incomplete or wrong-version
# install falls through and is replaced from the verified release archive.
if [[ -x "${BINARY}" ]] &&
  reports_pinned_version "${BINARY}" &&
  [[ -x "${INSTALL_PREFIX}/bin/ld.mold" ]] &&
  reports_pinned_version "${INSTALL_PREFIX}/bin/ld.mold" &&
  [[ -f "${INSTALL_PREFIX}/lib/mold/mold-wrapper.so" ]] &&
  [[ -e "${INSTALL_PREFIX}/libexec/mold/ld" ]]; then
  echo "mold ${MOLD_VERSION} already installed at ${BINARY}"
  exit 0
fi

EXPECTED_SHA="$(pinned_expected_sha "${ROOT_DIR}/tools/mold/SHA256SUMS" "${ARCHIVE}")"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT

pinned_fetch_verified "${URL}" "${TMP_DIR}/${ARCHIVE}" "${EXPECTED_SHA}"
tar -tzf "${TMP_DIR}/${ARCHIVE}" >/dev/null ||
  pinned_die "Invalid mold archive: ${ARCHIVE}"

STAGE_DIR="${TMP_DIR}/mold"
mkdir -p "${STAGE_DIR}"
tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "${STAGE_DIR}" --strip-components=1
[[ -x "${STAGE_DIR}/bin/mold" ]] ||
  pinned_die "Mold archive does not contain an executable bin/mold"
# Checked before the copy as well as after it: without this, a truncated
# archive that still matched its checksum would populate the prefix and only
# then fail on the version probe, leaving a half-installed tree behind.
[[ -x "${STAGE_DIR}/bin/ld.mold" ]] ||
  pinned_die "Mold archive does not contain an executable bin/ld.mold"
[[ -f "${STAGE_DIR}/lib/mold/mold-wrapper.so" ]] ||
  pinned_die "Mold archive is missing lib/mold/mold-wrapper.so"
[[ -e "${STAGE_DIR}/libexec/mold/ld" ]] ||
  pinned_die "Mold archive is missing libexec/mold/ld"

# Probe the staged binaries before touching the prefix. Checking only after
# the copy would leave a wrong-version tree installed, and the next run would
# then reuse it; failing here keeps a rejected archive out of the prefix
# entirely.
reports_pinned_version "${STAGE_DIR}/bin/mold" ||
  pinned_die "Staged mold does not report ${MOLD_VERSION}"
reports_pinned_version "${STAGE_DIR}/bin/ld.mold" ||
  pinned_die "Staged ld.mold does not report ${MOLD_VERSION}"

mkdir -p "${INSTALL_PREFIX}"
cp -a "${STAGE_DIR}/." "${INSTALL_PREFIX}/"

echo "Installed mold ${MOLD_VERSION} at ${BINARY}"
