# shellcheck shell=bash
#
# Shared helpers for installing pinned, checksum-verified tool archives.
#
# CI must never build a tool from source, so every installer in this
# repository resolves a pinned version from a `tools/<tool>/VERSION` file,
# looks the archive's SHA-256 up in the sibling `tools/<tool>/SHA256SUMS`
# file, and refuses to continue when either the pin or the digest is
# missing or does not match. Source this file; do not execute it.

pinned_die() {
  echo "$*" >&2
  exit 1
}

# Read a pinned version from a `tools/<tool>/VERSION` file.
pinned_version() {
  local version_file="$1" version
  [[ -f "${version_file}" ]] || pinned_die "Missing version file: ${version_file}"
  version="$(tr -d '[:space:]' <"${version_file}")"
  [[ -n "${version}" ]] || pinned_die "Empty version file: ${version_file}"
  printf '%s\n' "${version}"
}

# Look an archive's expected SHA-256 up in a `sha256sum`-format manifest.
pinned_expected_sha() {
  local checksum_file="$1" archive="$2" expected
  [[ -f "${checksum_file}" ]] || pinned_die "Missing checksum file: ${checksum_file}"
  expected="$(
    awk -v archive="${archive}" '$2 == archive || $2 == "*" archive {print $1; exit}' \
      "${checksum_file}"
  )"
  [[ -n "${expected}" ]] ||
    pinned_die "Missing SHA-256 for ${archive} in ${checksum_file}"
  printf '%s\n' "${expected}"
}

# Print the SHA-256 of a file using whichever digest tool the image provides.
pinned_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${path}" | awk '{print $1}'
  else
    pinned_die "Missing SHA-256 tool (sha256sum or shasum)."
  fi
}

# Download a URL to a destination and fail closed unless it matches its pin.
pinned_fetch_verified() {
  local url="$1" destination="$2" expected="$3" actual
  curl -sSfL "${url}" -o "${destination}" ||
    pinned_die "Failed to download ${url}"
  actual="$(pinned_sha256 "${destination}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${url}." >&2
    echo "Expected: ${expected}" >&2
    echo "Actual:   ${actual}" >&2
    exit 1
  fi
}
