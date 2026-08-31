#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_BIN="${HF_BIN:-${ROOT_DIR}/.venv/bin/hf}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
LORA_REPO="${LORA_REPO:-larryvrh/MiniMax-H3-Turbo-Lora}"
LORA_REVISION="${LORA_REVISION:-43a74557ac3f6539db8e0f2a959d03feb7a81480}"
LORA_NAME="${LORA_NAME:-minimax_h3_turbo_v4_step600_ema.safetensors}"
LORA_DIR="${LORA_DIR:-${ROOT_DIR}/models/loras}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -x "${HF_BIN}" ]]; then
  echo "Hugging Face CLI not found: ${HF_BIN}. Run ./setup.sh first." >&2
  exit 1
fi

mkdir -p "${LORA_DIR}"
"${HF_BIN}" download "${LORA_REPO}" "${LORA_NAME}" \
  --revision "${LORA_REVISION}" \
  --local-dir "${LORA_DIR}"

"${PYTHON_BIN}" - "${LORA_DIR}/${LORA_NAME}" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

path = Path(sys.argv[1])
with safe_open(path, framework="numpy") as handle:
    keys = list(handle.keys())
    metadata = handle.metadata()
print(json.dumps({"path": str(path), "bytes": path.stat().st_size, "tensors": len(keys), "metadata": metadata}))
PY
