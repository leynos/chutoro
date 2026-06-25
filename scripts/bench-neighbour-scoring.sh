#!/usr/bin/env bash
# Run the neighbour-scoring benchmark binary through hyperfine.

set -euo pipefail

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

require_command cargo
require_command jq
require_command hyperfine

branch_name="$(git branch --show-current 2>/dev/null || printf 'unknown')"
branch_name="${branch_name//\//-}"
log_dir="${TMPDIR:-/tmp}/chutoro-benches-${UID}"
mkdir -p "${log_dir}"
if [[ ! -O "${log_dir}" ]]; then
  printf 'log directory is not owned by the current user: %s\n' "${log_dir}" >&2
  exit 1
fi
chmod 700 "${log_dir}"
log_path="${log_dir}/hyperfine-neighbour-scoring-${branch_name}.out"

if ! build_messages="$(
  cargo bench -p chutoro-benches --bench neighbour_scoring --no-run --message-format=json
)"; then
  printf 'cargo bench failed to build the neighbour_scoring benchmark\n' >&2
  exit 1
fi

bench_binary="$(
  printf '%s\n' "${build_messages}" |
    jq -r '
      try (
        select(
          .reason == "compiler-artifact"
          and .target.name == "neighbour_scoring"
          and .executable != null
        ) | .executable
      ) catch empty
    ' |
    tail -n 1
)"

if [[ -z "${bench_binary}" ]]; then
  printf 'neighbour_scoring benchmark binary was not found\n' >&2
  exit 1
fi

escaped_bench_binary="$(printf '%q' "${bench_binary}")"

hyperfine \
  --shell bash \
  --warmup 1 \
  --runs 10 \
  "${escaped_bench_binary} --bench --profile-time 1" \
  2>&1 | tee "${log_path}"
