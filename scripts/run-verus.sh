#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROOF_FILE="${VERUS_PROOF_FILE:-${ROOT_DIR}/verus/edge_harvest_proofs.rs}"
VERSION_FILE="${ROOT_DIR}/tools/verus/VERSION"

if [[ ! -f "${VERSION_FILE}" ]]; then
  echo "Missing Verus version file: ${VERSION_FILE}" >&2
  exit 1
fi

VERUS_VERSION="$(cat "${VERSION_FILE}")"
DEFAULT_INSTALL_DIR="${VERUS_INSTALL_DIR:-${ROOT_DIR}/.verus/${VERUS_VERSION}}"
DEFAULT_VERUS_BIN="${DEFAULT_INSTALL_DIR}/verus/verus"
VERUS_BIN="${VERUS_BIN:-${DEFAULT_VERUS_BIN}}"

resolve_verus_bin() {
  local candidate="$1"

  if [[ -z "${candidate}" ]]; then
    return 1
  fi

  if [[ -f "${candidate}" && -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  if [[ -d "${candidate}" ]]; then
    return 1
  fi

  if command -v "${candidate}" >/dev/null 2>&1; then
    command -v "${candidate}"
    return 0
  fi

  return 1
}

RESOLVED_VERUS_BIN="$(resolve_verus_bin "${VERUS_BIN}" || true)"
if [[ -n "${RESOLVED_VERUS_BIN}" ]]; then
  VERUS_BIN="${RESOLVED_VERUS_BIN}"
else
  if [[ "${VERUS_BIN}" != "${DEFAULT_VERUS_BIN}" ]]; then
    echo "VERUS_BIN is not executable: ${VERUS_BIN}" >&2
    echo "Falling back to ${DEFAULT_VERUS_BIN}" >&2
    VERUS_BIN="${DEFAULT_VERUS_BIN}"
  fi

  if [[ -z "$(resolve_verus_bin "${VERUS_BIN}" || true)" ]]; then
    VERUS_INSTALL_DIR="${DEFAULT_INSTALL_DIR}" scripts/install-verus.sh
  fi

  RESOLVED_VERUS_BIN="$(resolve_verus_bin "${VERUS_BIN}" || true)"
  if [[ -n "${RESOLVED_VERUS_BIN}" ]]; then
    VERUS_BIN="${RESOLVED_VERUS_BIN}"
  fi
fi

if [[ ! -f "${VERUS_BIN}" || ! -x "${VERUS_BIN}" ]]; then
  echo "Verus binary not found after install: ${VERUS_BIN}" >&2
  exit 1
fi

if [[ ! -f "${PROOF_FILE}" ]]; then
  echo "Verus proof file not found: ${PROOF_FILE}" >&2
  exit 1
fi

VERUS_VERSION_OUTPUT="$("${VERUS_BIN}" --version 2>&1)"
TOOLCHAIN="$(echo "${VERUS_VERSION_OUTPUT}" | awk -F ':' '/Toolchain:/ {gsub(/^[ \t]+/, "", $2); print $2}')"

if [[ -z "${TOOLCHAIN}" ]]; then
  echo "Failed to parse Verus toolchain from output:" >&2
  echo "${VERUS_VERSION_OUTPUT}" >&2
  exit 1
fi

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup is required to install toolchain ${TOOLCHAIN}" >&2
  exit 1
fi

if ! rustup which --toolchain "${TOOLCHAIN}" rustc >/dev/null 2>&1; then
  rustup toolchain install "${TOOLCHAIN}"
fi

"${VERUS_BIN}" "${PROOF_FILE}"
