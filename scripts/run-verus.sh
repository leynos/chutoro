#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROOF_FILE="${VERUS_PROOF_FILE:-${ROOT_DIR}/verus/edge_harvest_proofs.rs}"
VERUS_BIN="${VERUS_BIN:-verus}"

if [[ -x "${VERUS_BIN}" ]]; then
  VERUS_CMD="${VERUS_BIN}"
else
  VERUS_CMD="$(command -v "${VERUS_BIN}" || true)"
fi

if [[ -z "${VERUS_CMD}" ]]; then
  echo "Verus binary not found. Set VERUS_BIN or install per docs/verus-toolchain.md" >&2
  exit 1
fi

if [[ ! -f "${PROOF_FILE}" ]]; then
  echo "Verus proof file not found: ${PROOF_FILE}" >&2
  exit 1
fi

"${VERUS_CMD}" "${PROOF_FILE}"
