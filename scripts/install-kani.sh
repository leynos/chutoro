#!/usr/bin/env bash
#
# Install the pinned Kani model checker without compiling anything.
#
# Kani is a two-part tool and both parts must be installed together:
#
#   * the `cargo-kani`/`kani` front-end, published as a Cargo QuickInstall
#     binary archive; and
#   * the CBMC-backed verifier bundle, published on the Kani release.
#
# Extracting only the bundle leaves `cargo kani` unavailable, and
# installing only the front-end leaves the verifier missing, so both
# archives are pinned in tools/kani/SHA256SUMS at the same version. The
# front-end embeds its own version when it resolves the bundle, so the two
# pins must always be bumped together.
#
# `cargo kani setup` installs its pinned nightly toolchain through rustup.
# That toolchain is directed at a Kani-specific RUSTUP_HOME so it never
# shares ownership with the job's ordinary Rust toolchain cache, and so the
# three Kani cache paths (the front-end in Cargo home, KANI_HOME, and the
# Kani RUSTUP_HOME) can be restored as one unit.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/pinned-download.sh
source "${ROOT_DIR}/scripts/lib/pinned-download.sh"

KANI_VERSION="$(pinned_version "${ROOT_DIR}/tools/kani/VERSION")"
KANI_TARGET="${KANI_TARGET:-x86_64-unknown-linux-gnu}"
CHECKSUM_FILE="${ROOT_DIR}/tools/kani/SHA256SUMS"

CARGO_BIN_DIR="${KANI_CARGO_BIN_DIR:-${HOME}/.cargo/bin}"
KANI_HOME="${KANI_HOME:-${HOME}/.kani}"
KANI_RUSTUP_HOME="${KANI_RUSTUP_HOME:-${HOME}/.kani-rustup}"
export KANI_HOME

FRONTEND_BINARY="${CARGO_BIN_DIR}/cargo-kani"
VERIFIER_BINARY="${KANI_HOME}/kani-${KANI_VERSION}/bin/kani-compiler"
TOOLCHAIN_DIR="${KANI_HOME}/kani-${KANI_VERSION}/toolchain"

# Executable probe. A warm cache must repeat neither download, but a
# partially restored installation is reinstalled rather than trusted.
if [[ -x "${FRONTEND_BINARY}" && -x "${VERIFIER_BINARY}" && -d "${TOOLCHAIN_DIR}" ]]; then
  echo "Kani ${KANI_VERSION} already installed (KANI_HOME=${KANI_HOME})"
  exit 0
fi

FRONTEND_ARCHIVE="kani-verifier-${KANI_VERSION}-${KANI_TARGET}.tar.gz"
FRONTEND_URL="https://github.com/cargo-bins/cargo-quickinstall/releases/download/kani-verifier-${KANI_VERSION}/${FRONTEND_ARCHIVE}"
BUNDLE_ARCHIVE="kani-${KANI_VERSION}-${KANI_TARGET}.tar.gz"
BUNDLE_URL="https://github.com/model-checking/kani/releases/download/kani-${KANI_VERSION}/${BUNDLE_ARCHIVE}"

FRONTEND_SHA="$(pinned_expected_sha "${CHECKSUM_FILE}" "${FRONTEND_ARCHIVE}")"
BUNDLE_SHA="$(pinned_expected_sha "${CHECKSUM_FILE}" "${BUNDLE_ARCHIVE}")"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT

pinned_fetch_verified "${FRONTEND_URL}" "${TMP_DIR}/${FRONTEND_ARCHIVE}" "${FRONTEND_SHA}"
pinned_fetch_verified "${BUNDLE_URL}" "${TMP_DIR}/${BUNDLE_ARCHIVE}" "${BUNDLE_SHA}"

mkdir -p "${CARGO_BIN_DIR}" "${KANI_RUSTUP_HOME}"
tar -xzf "${TMP_DIR}/${FRONTEND_ARCHIVE}" -C "${TMP_DIR}" cargo-kani kani
chmod +x "${TMP_DIR}/cargo-kani" "${TMP_DIR}/kani"
mv -f "${TMP_DIR}/cargo-kani" "${FRONTEND_BINARY}"
mv -f "${TMP_DIR}/kani" "${CARGO_BIN_DIR}/kani"

# `setup` unpacks the verified local bundle instead of downloading one.
RUSTUP_HOME="${KANI_RUSTUP_HOME}" \
  "${CARGO_BIN_DIR}/kani" setup --use-local-bundle "${TMP_DIR}/${BUNDLE_ARCHIVE}"

for path in "${FRONTEND_BINARY}" "${VERIFIER_BINARY}"; do
  [[ -x "${path}" ]] || pinned_die "Kani installation left ${path} missing"
done
[[ -d "${TOOLCHAIN_DIR}" ]] ||
  pinned_die "Kani installation left ${TOOLCHAIN_DIR} missing"

echo "Installed Kani ${KANI_VERSION} (KANI_HOME=${KANI_HOME})"
