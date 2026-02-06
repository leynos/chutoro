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

  if [[ -d "${candidate}" ]]; then
    local bin
    for bin in "${candidate}/verus" "${candidate}/verus/verus" "${candidate}/bin/verus"; do
      if [[ -f "${bin}" && -x "${bin}" ]]; then
        echo "${bin}"
        return 0
      fi
    done
    return 1
  fi

  if [[ -f "${candidate}" && -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  if command -v "${candidate}" >/dev/null 2>&1; then
    command -v "${candidate}"
    return 0
  fi

  return 1
}

ensure_toolchain_installed() {
  local toolchain="$1"

  if ! command -v rustup >/dev/null 2>&1; then
    echo "rustup is required to install toolchain ${toolchain}" >&2
    exit 1
  fi

  if ! rustup which --toolchain "${toolchain}" rustc >/dev/null 2>&1; then
    rustup toolchain install "${toolchain}"
  fi
}

# parse_verus_toolchain handles "Toolchain: <name>" and
# "required rust toolchain <name>" outputs from verus --version.
parse_verus_toolchain() {
  local output="$1"

  local toolchain
  toolchain="$(echo "${output}" | awk -F ':' '/Toolchain:/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
  if [[ -n "${toolchain}" ]]; then
    echo "${toolchain}"
    return 0
  fi

  toolchain="$(
    echo "${output}" | awk '/required rust toolchain/ {for (i = 1; i <= NF; i++) if ($i == "toolchain") {print $(i + 1); exit}}'
  )"
  if [[ -n "${toolchain}" ]]; then
    echo "${toolchain}"
    return 0
  fi

  return 1
}

collect_verus_version_output() {
  VERUS_VERSION_STATUS=0
  VERUS_VERSION_OUTPUT="$("${VERUS_BIN}" --version 2>&1)" || VERUS_VERSION_STATUS=$?
}

resolve_verus_toolchain() {
  local output="$1"
  local toolchain

  toolchain="$(parse_verus_toolchain "${output}" || true)"
  if [[ -z "${toolchain}" ]]; then
    echo "Failed to parse Verus toolchain from output:" >&2
    echo "${output}" >&2
    exit 1
  fi

  echo "${toolchain}"
}

ensure_verus_toolchain() {
  local toolchain

  collect_verus_version_output
  toolchain="$(resolve_verus_toolchain "${VERUS_VERSION_OUTPUT}")"

  ensure_toolchain_installed "${toolchain}"

  if [[ "${VERUS_VERSION_STATUS}" -ne 0 ]]; then
    collect_verus_version_output
    if [[ "${VERUS_VERSION_STATUS}" -ne 0 ]]; then
      echo "Failed to run ${VERUS_BIN} --version after installing toolchain." >&2
      echo "${VERUS_VERSION_OUTPUT}" >&2
      exit 1
    fi
  fi

  TOOLCHAIN="${toolchain}"
}

RESOLVED_VERUS_BIN="$(resolve_verus_bin "${VERUS_BIN}" || true)"
if [[ -n "${RESOLVED_VERUS_BIN}" ]]; then
  VERUS_BIN="${RESOLVED_VERUS_BIN}"
else
  if [[ "${VERUS_BIN}" != "${DEFAULT_VERUS_BIN}" ]]; then
    if [[ -d "${VERUS_BIN}" ]]; then
      echo "VERUS_BIN directory contains no recognised Verus binary: ${VERUS_BIN}" >&2
    else
      echo "VERUS_BIN is not executable: ${VERUS_BIN}" >&2
    fi
    echo "Falling back to ${DEFAULT_VERUS_BIN}" >&2
    VERUS_BIN="${DEFAULT_VERUS_BIN}"
  fi

  if [[ -z "$(resolve_verus_bin "${VERUS_BIN}" || true)" ]]; then
    VERUS_INSTALL_DIR="${DEFAULT_INSTALL_DIR}" "${ROOT_DIR}/scripts/install-verus.sh"
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

ensure_verus_toolchain

if "${VERUS_BIN}" "${PROOF_FILE}"; then
  exit 0
else
  # $? captures the exit status from the Verus invocation above.
  status=$?
  echo "Verus proofs failed (exit ${status})." >&2
  echo "Binary: ${VERUS_BIN}" >&2
  echo "Proof file: ${PROOF_FILE}" >&2
  echo "Toolchain: ${TOOLCHAIN}" >&2
  exit "${status}"
fi
