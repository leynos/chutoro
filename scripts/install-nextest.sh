#!/usr/bin/env bash
#
# Install the pinned cargo-nextest release binary.
#
# nextest publishes a versioned archive per target, so CI downloads that
# archive directly rather than resolving a binary through `cargo binstall`
# or compiling the crate. The archive's SHA-256 is pinned in
# tools/nextest/SHA256SUMS and a mismatch is a hard error.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/pinned-download.sh
source "${ROOT_DIR}/scripts/lib/pinned-download.sh"

NEXTEST_VERSION="$(pinned_version "${ROOT_DIR}/tools/nextest/VERSION")"
NEXTEST_TARGET="${NEXTEST_TARGET:-x86_64-unknown-linux-gnu}"
INSTALL_DIR="${NEXTEST_INSTALL_DIR:-${HOME}/.cargo/bin}"
BINARY="${INSTALL_DIR}/cargo-nextest"
ARCHIVE="cargo-nextest-${NEXTEST_VERSION}-${NEXTEST_TARGET}.tar.gz"
URL="https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-${NEXTEST_VERSION}/${ARCHIVE}"

# Executable probe: a warm cache must not repeat the download, but a
# restored file that cannot report the pinned version is replaced.
if [[ -x "${BINARY}" ]] &&
  "${BINARY}" --version 2>/dev/null | grep -qF "cargo-nextest ${NEXTEST_VERSION} "; then
  echo "cargo-nextest ${NEXTEST_VERSION} already installed at ${BINARY}"
  exit 0
fi

EXPECTED_SHA="$(pinned_expected_sha "${ROOT_DIR}/tools/nextest/SHA256SUMS" "${ARCHIVE}")"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT

pinned_fetch_verified "${URL}" "${TMP_DIR}/${ARCHIVE}" "${EXPECTED_SHA}"

tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "${TMP_DIR}" cargo-nextest
chmod +x "${TMP_DIR}/cargo-nextest"
mkdir -p "${INSTALL_DIR}"
mv -f "${TMP_DIR}/cargo-nextest" "${BINARY}"

"${BINARY}" --version | grep -qF "cargo-nextest ${NEXTEST_VERSION} " ||
  pinned_die "Installed cargo-nextest does not report ${NEXTEST_VERSION}"
echo "Installed cargo-nextest ${NEXTEST_VERSION} at ${BINARY}"
