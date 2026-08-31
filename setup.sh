#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
DIFFUSERS_COMMIT="${DIFFUSERS_COMMIT:-1b98ae1060b765f2efe22540f52691b8c00a83f1}"
DIFFUSERS_SOURCE="${DIFFUSERS_SOURCE:-/tmp/diffusers-h3}"
KNOWN_PIP_WARNINGS='^(torch-mlu-ops .* requires torch-mlu|.*torch[-_]?mlu.* has requirement pandas)'

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0

if [[ "${BASE_PYTHON}" == */* ]]; then
  BASE_PYTHON_BIN="${BASE_PYTHON}"
else
  BASE_PYTHON_BIN="$(command -v "${BASE_PYTHON}" || true)"
fi
if [[ -z "${BASE_PYTHON_BIN}" || ! -x "${BASE_PYTHON_BIN}" ]]; then
  echo "Base Python not found: ${BASE_PYTHON}" >&2
  exit 1
fi

"${BASE_PYTHON_BIN}" - <<'PY'
import torch
import torch_mlu

if not torch.mlu.is_available():
    raise SystemExit("torch_mlu imported, but no MLU device is available")
print(f"Base runtime: torch={torch.__version__}, device={torch.mlu.get_device_name(0)}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --requirement "${ROOT_DIR}/requirements.txt"

if [[ -d "${DIFFUSERS_SOURCE}/.git" ]] \
  && [[ "$(git -C "${DIFFUSERS_SOURCE}" rev-parse HEAD)" == "${DIFFUSERS_COMMIT}" ]]; then
  "${VENV_DIR}/bin/python" -m pip install --force-reinstall --no-deps "${DIFFUSERS_SOURCE}"
else
  "${VENV_DIR}/bin/python" -m pip install --force-reinstall --no-deps \
    "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_COMMIT}"
fi

PIP_CHECK_OUTPUT="$("${VENV_DIR}/bin/python" -m pip check 2>&1 || true)"
UNEXPECTED_PIP_ERRORS="$(
  printf '%s\n' "${PIP_CHECK_OUTPUT}" \
    | grep -v -E "${KNOWN_PIP_WARNINGS}" \
    | grep -v '^$' \
    || true
)"
if [[ -n "${UNEXPECTED_PIP_ERRORS}" ]] && [[ "${UNEXPECTED_PIP_ERRORS}" != "No broken requirements found." ]]; then
  printf '%s\n' "${UNEXPECTED_PIP_ERRORS}" >&2
  exit 1
fi
printf '%s\n' "${PIP_CHECK_OUTPUT}" \
  | grep -E "${KNOWN_PIP_WARNINGS}" \
  || true

"${VENV_DIR}/bin/python" - <<'PY'
import accelerate
import diffusers
import huggingface_hub
import peft
import torch
import torch_mlu
import transformers
from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline

assert torch.mlu.is_available()
print(f"diffusers={diffusers.__version__}")
print(f"transformers={transformers.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"huggingface_hub={huggingface_hub.__version__}")
print(f"peft={peft.__version__}")
print("MiniMax-H3 imports: OK")
PY
