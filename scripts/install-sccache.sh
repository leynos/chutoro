#!/usr/bin/env bash
#
# Install the pinned sccache release binary.
#
# The shared `setup-rust` action can install sccache through
# mozilla-actions/sccache-action, but a server started inside an action step
# binds GitHub's v2 cache service because the runner re-injects the reserved
# cache variables into action steps. Every job here therefore sets
# `use-sccache: 'false'` and installs the binary from the pinned release
# archive instead, so a later `run:` step owns the server and its backend.
# The archive's SHA-256 is pinned in tools/sccache/SHA256SUMS and a mismatch
# is a hard error.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/pinned-download.sh
source "${ROOT_DIR}/scripts/lib/pinned-download.sh"

SCCACHE_VERSION="$(pinned_version "${ROOT_DIR}/tools/sccache/VERSION")"
# The musl build is deliberate: it has no glibc floor, so the same pinned
# archive runs on GitHub's and Ubicloud's images without a per-image matrix.
SCCACHE_TARGET="${SCCACHE_TARGET:-x86_64-unknown-linux-musl}"
INSTALL_DIR="${SCCACHE_INSTALL_DIR:-${HOME}/.cargo/bin}"
BINARY="${INSTALL_DIR}/sccache"
STEM="sccache-v${SCCACHE_VERSION}-${SCCACHE_TARGET}"
ARCHIVE="${STEM}.tar.gz"
URL="https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/${ARCHIVE}"

# Executable probe: a warm image must not repeat the download, but a file
# that cannot report the pinned version is replaced.
if [[ -x "${BINARY}" ]] &&
  "${BINARY}" --version 2>/dev/null | grep -qFx "sccache ${SCCACHE_VERSION}"; then
  echo "sccache ${SCCACHE_VERSION} already installed at ${BINARY}"
  exit 0
fi

EXPECTED_SHA="$(pinned_expected_sha "${ROOT_DIR}/tools/sccache/SHA256SUMS" "${ARCHIVE}")"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT

pinned_fetch_verified "${URL}" "${TMP_DIR}/${ARCHIVE}" "${EXPECTED_SHA}"

tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "${TMP_DIR}" "${STEM}/sccache"
chmod +x "${TMP_DIR}/${STEM}/sccache"
mkdir -p "${INSTALL_DIR}"
mv -f "${TMP_DIR}/${STEM}/sccache" "${BINARY}"

"${BINARY}" --version | grep -qFx "sccache ${SCCACHE_VERSION}" ||
  pinned_die "Installed sccache does not report ${SCCACHE_VERSION}"
echo "Installed sccache ${SCCACHE_VERSION} at ${BINARY}"
