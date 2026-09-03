#!/usr/bin/env bash
#
# Install the pinned Verus release archive.
#
# Verus publishes a prebuilt archive per target, so CI never builds it from
# source. The archive's SHA-256 is pinned in tools/verus/SHA256SUMS and a
# mismatch is a hard error.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/pinned-download.sh
source "${ROOT_DIR}/scripts/lib/pinned-download.sh"

VERUS_VERSION="$(pinned_version "${ROOT_DIR}/tools/verus/VERSION")"
VERUS_TARGET="${VERUS_TARGET:-x86-linux}"
INSTALL_DIR="${VERUS_INSTALL_DIR:-${ROOT_DIR}/.verus/${VERUS_VERSION}}"
ARCHIVE="verus-${VERUS_VERSION}-${VERUS_TARGET}.zip"
URL="https://github.com/verus-lang/verus/releases/download/release/${VERUS_VERSION}/${ARCHIVE}"

# Executable probe: a warm cache must not repeat the download.
if [[ -x "${INSTALL_DIR}/verus/verus" ]]; then
  echo "Verus ${VERUS_VERSION} already installed at ${INSTALL_DIR}/verus"
  exit 0
fi

EXPECTED_SHA="$(pinned_expected_sha "${ROOT_DIR}/tools/verus/SHA256SUMS" "${ARCHIVE}")"

mkdir -p "${INSTALL_DIR}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT

pinned_fetch_verified "${URL}" "${TMP_DIR}/${ARCHIVE}" "${EXPECTED_SHA}"

unzip -q "${TMP_DIR}/${ARCHIVE}" -d "${INSTALL_DIR}"

EXTRACTED_DIR="${INSTALL_DIR}/verus-${VERUS_TARGET}"
if [[ ! -d "${EXTRACTED_DIR}" ]]; then
  EXTRACTED_DIR="$(find "${INSTALL_DIR}" -maxdepth 1 -type d -name 'verus-*' | head -n 1)"
fi

if [[ -z "${EXTRACTED_DIR}" || ! -d "${EXTRACTED_DIR}" ]]; then
  pinned_die "Unable to locate extracted Verus directory under ${INSTALL_DIR}"
fi

rm -rf -- "${INSTALL_DIR}/verus"
mv "${EXTRACTED_DIR}" "${INSTALL_DIR}/verus"

[[ -x "${INSTALL_DIR}/verus/verus" ]] ||
  pinned_die "Verus installation left ${INSTALL_DIR}/verus/verus missing"

cat <<EOM
Installed Verus ${VERUS_VERSION} in ${INSTALL_DIR}/verus
Export VERUS_BIN=${INSTALL_DIR}/verus/verus
EOM
