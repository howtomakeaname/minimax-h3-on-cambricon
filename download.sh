#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_BIN="${HF_BIN:-${ROOT_DIR}/.venv/bin/hf}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL_ID="${MODEL_ID:-MiniMaxAI/MiniMax-H3}"
MODEL_REVISION="${MODEL_REVISION:-42ed227ee7df40d41602854ae760620d6eb651fe}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/MiniMax-H3-diffusers}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -x "${HF_BIN}" ]]; then
  echo "Hugging Face CLI not found: ${HF_BIN}. Run ./setup.sh first." >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"

"${HF_BIN}" download "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --local-dir "${MODEL_DIR}" \
  --include "model_index.json" \
  --include "modular_model_index.json" \
  --include "LICENSE" \
  --include "README.md" \
  --include "audio_scheduler/*" \
  --include "audio_vae/*" \
  --include "processor/*" \
  --include "scheduler/*" \
  --include "text_encoder/*" \
  --include "tokenizer/*" \
  --include "transformer/*" \
  --include "vae/*"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/verify_model.py" "${MODEL_DIR}"
echo "MiniMax-H3 Diffusers FL2VA files are available at ${MODEL_DIR}"
