#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="${ROOT_DIR}/tools/verus/VERSION"

if [[ ! -f "${VERSION_FILE}" ]]; then
  echo "Missing Verus version file: ${VERSION_FILE}" >&2
  exit 1
fi

VERUS_VERSION="$(cat "${VERSION_FILE}")"
VERUS_TARGET="${VERUS_TARGET:-x86-linux}"
INSTALL_DIR="${VERUS_INSTALL_DIR:-${ROOT_DIR}/.verus/${VERUS_VERSION}}"
ARCHIVE="verus-${VERUS_VERSION}-${VERUS_TARGET}.zip"
URL="https://github.com/verus-lang/verus/releases/download/release/${VERUS_VERSION}/${ARCHIVE}"

if [[ -x "${INSTALL_DIR}/verus/verus" ]]; then
  echo "Verus ${VERUS_VERSION} already installed at ${INSTALL_DIR}/verus"
  exit 0
fi

mkdir -p "${INSTALL_DIR}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

curl -sSfL "${URL}" -o "${TMP_DIR}/${ARCHIVE}"
unzip -q "${TMP_DIR}/${ARCHIVE}" -d "${INSTALL_DIR}"

EXTRACTED_DIR="${INSTALL_DIR}/verus-${VERUS_TARGET}"
if [[ ! -d "${EXTRACTED_DIR}" ]]; then
  EXTRACTED_DIR="$(find "${INSTALL_DIR}" -maxdepth 1 -type d -name 'verus-*' | head -n 1)"
fi

if [[ -z "${EXTRACTED_DIR}" || ! -d "${EXTRACTED_DIR}" ]]; then
  echo "Unable to locate extracted Verus directory under ${INSTALL_DIR}" >&2
  exit 1
fi

rm -rf "${INSTALL_DIR}/verus"
mv "${EXTRACTED_DIR}" "${INSTALL_DIR}/verus"

cat <<EOM
Installed Verus ${VERUS_VERSION} in ${INSTALL_DIR}/verus
Export VERUS_BIN=${INSTALL_DIR}/verus/verus
EOM
