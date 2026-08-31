#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

ln -s "${ROOT_DIR}/tests/fixtures/cnmon" "${TEMP_DIR}/cnmon"
ln -s "${ROOT_DIR}/tests/fixtures/numactl" "${TEMP_DIR}/numactl"

output="$(
  PATH="${TEMP_DIR}:${PATH}" \
  MLU_VISIBLE_DEVICES=5 \
  MINIMAX_H3_RUNNER=/usr/bin/true \
  "${ROOT_DIR}/tools/numa_run.sh" --device mlu:0 --steps 2 2>&1
)"

grep -q 'Binding MLU 0 (0000:69:03.0) to NUMA node 1.' <<<"${output}"
grep -q -- '--cpunodebind=1' <<<"${output}"
grep -q -- '--membind=1' <<<"${output}"
