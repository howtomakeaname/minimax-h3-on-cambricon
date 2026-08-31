#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0

if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  echo "Virtual environment not found. Run ./setup.sh first." >&2
  exit 1
fi

exec "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/infer.py" "$@"
