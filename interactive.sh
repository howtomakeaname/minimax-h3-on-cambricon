#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${MINIMAX_H3_RUNNER:-${ROOT_DIR}/run.sh}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./interactive.sh [infer.py options]

The script asks for generation parameters and then reads a multiline prompt.
Press Enter to accept each default. Enter ::end on its own line to generate.
Quality mode uses 19 model evaluations by default. Turbo is opt-in.

Additional infer.py options can be appended to the command line.
EOF
  exit 0
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

get_mlu_pids() {
  cnmon info -p 2>/dev/null \
    | awk -F: '/^[[:space:]]+PID[[:space:]]*:/ {
        gsub(/[[:space:]]/, "", $2)
        if ($2 ~ /^[0-9]+$/) print $2
      }'
}

check_existing_mlu_processes() {
  if [[ "${MINIMAX_H3_SKIP_PROCESS_CHECK:-0}" == "1" ]] || ! command -v cnmon >/dev/null 2>&1; then
    return
  fi

  local -a mlu_pids remaining_pids
  local pid pid_csv answer attempt
  mapfile -t mlu_pids < <(get_mlu_pids)
  if (( ${#mlu_pids[@]} == 0 )); then
    return
  fi

  printf '%s\n' "Active MLU processes detected:"
  pid_csv="$(printf '%s,' "${mlu_pids[@]}")"
  ps -ww -p "${pid_csv%,}" -o pid=,user=,etime=,rss=,args= || true
  read -r -p "Terminate these processes before starting? (y/N): " answer
  if [[ "${answer,,}" != "y" && "${answer,,}" != "yes" ]]; then
    printf '%s\n' "Generation cancelled; existing MLU processes were left untouched."
    exit 1
  fi

  for pid in "${mlu_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null && ! kill -TERM "${pid}" 2>/dev/null; then
      printf 'Could not terminate PID %s.\n' "${pid}" >&2
      exit 1
    fi
  done

  for attempt in {1..15}; do
    remaining_pids=()
    for pid in "${mlu_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        remaining_pids+=("${pid}")
      fi
    done
    if (( ${#remaining_pids[@]} == 0 )); then
      printf '%s\n' "Previous MLU processes stopped."
      return
    fi
    sleep 1
  done

  printf 'Processes still running after 15 seconds: %s\n' "${remaining_pids[*]}" >&2
  read -r -p "Force kill them? (y/N): " answer
  if [[ "${answer,,}" != "y" && "${answer,,}" != "yes" ]]; then
    printf '%s\n' "Generation cancelled; no force kill was sent."
    exit 1
  fi
  for pid in "${remaining_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}"
    fi
  done
  printf '%s\n' "Previous MLU processes force-killed."
}

check_existing_mlu_processes

read -r -p "Output filename [random]: " output_name
output_name="$(trim "${output_name}")"
if [[ -z "${output_name}" ]]; then
  random_suffix="$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
  output_name="minimax-h3-${random_suffix}.mp4"
fi

case "${output_name}" in
  */*|*\\*|"."|"..")
    printf '%s\n' "The filename must not contain a path." >&2
    exit 2
    ;;
esac
if [[ "${output_name,,}" != *.mp4 ]]; then
  output_name="${output_name%.*}.mp4"
fi

read -r -p "Workflow (t2va/fl2va) [t2va]: " workflow
workflow="${workflow:-t2va}"
if [[ "${workflow}" != "t2va" && "${workflow}" != "fl2va" ]]; then
  printf '%s\n' "Workflow must be either t2va or fl2va." >&2
  exit 2
fi

image=""
last_image=""
if [[ "${workflow}" == "fl2va" ]]; then
  read -r -p "First-frame image path [none]: " image
  read -r -p "Last-frame image path [none]: " last_image
  image="$(trim "${image}")"
  last_image="$(trim "${last_image}")"
  if [[ -z "${image}" && -z "${last_image}" ]]; then
    printf '%s\n' "FL2VA requires a first-frame image, a last-frame image, or both." >&2
    exit 2
  fi
  if [[ -n "${image}" && ! -f "${image}" ]]; then
    printf 'First-frame image does not exist: %s\n' "${image}" >&2
    exit 2
  fi
  if [[ -n "${last_image}" && ! -f "${last_image}" ]]; then
    printf 'Last-frame image does not exist: %s\n' "${last_image}" >&2
    exit 2
  fi
fi

read -r -p "Performance mode (quality/turbo) [quality]: " performance_mode
performance_mode="${performance_mode:-quality}"
if [[ "${performance_mode}" != "turbo" && "${performance_mode}" != "quality" ]]; then
  printf '%s\n' "Performance mode must be either turbo or quality." >&2
  exit 2
fi

turbo_lora="${TURBO_LORA:-${ROOT_DIR}/models/loras/minimax_h3_turbo_v4_step600_ema.safetensors}"
if [[ "${performance_mode}" == "turbo" ]]; then
  default_steps=9
  if [[ ! -f "${turbo_lora}" ]]; then
    printf 'Turbo LoRA does not exist: %s. Run ./download_turbo.sh first.\n' "${turbo_lora}" >&2
    exit 2
  fi
else
  default_steps=20
fi

read -r -p "Width [1344]: " width
read -r -p "Height [768]: " height
read -r -p "Number of frames [124]: " num_frames
read -r -p "Scheduler points [${default_steps}]: " steps
read -r -p "Seed [42]: " seed
read -r -p "Attention (flash/default) [flash]: " attention
if [[ "${performance_mode}" == "turbo" ]]; then
  read -r -p "Turbo LoRA strength [1.0]: " lora_scale
  lora_scale="${lora_scale:-1.0}"
fi
read -r -p "Offload reserve margin [12GB]: " offload_margin
read -r -p "Offload mode (copyback/cpu-master/pinned-master) [cpu-master]: " offload_mode
read -r -p "Device [mlu:0]: " device
read -r -p "Model directory [${ROOT_DIR}/models/MiniMax-H3-diffusers]: " model_dir

width="${width:-1344}"
height="${height:-768}"
num_frames="${num_frames:-124}"
steps="${steps:-${default_steps}}"
seed="${seed:-42}"
attention="${attention:-flash}"
offload_margin="${offload_margin:-12GB}"
offload_mode="${offload_mode:-cpu-master}"
device="${device:-mlu:0}"
model_dir="${model_dir:-${ROOT_DIR}/models/MiniMax-H3-diffusers}"

if [[ ! "${width}" =~ ^[1-9][0-9]*$ ]] || (( width % 32 != 0 )); then
  printf '%s\n' "Width must be a positive multiple of 32." >&2
  exit 2
fi
if [[ ! "${height}" =~ ^[1-9][0-9]*$ ]] || (( height % 32 != 0 )); then
  printf '%s\n' "Height must be a positive multiple of 32." >&2
  exit 2
fi
if [[ ! "${num_frames}" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' "Number of frames must be a positive integer." >&2
  exit 2
fi

aligned_num_frames="${num_frames}"
while (( aligned_num_frames % 17 != 5 )); do
  aligned_num_frames=$((aligned_num_frames + 1))
done
if (( aligned_num_frames < 120 || aligned_num_frames > 360 )); then
  printf 'Frame count %s would align to %s; the aligned value must cover 5 to 15 seconds.\n' \
    "${num_frames}" \
    "${aligned_num_frames}" >&2
  exit 2
fi
if [[ ! "${steps}" =~ ^[0-9]+$ ]] || (( steps < 2 )); then
  printf '%s\n' "Scheduler points must be an integer of at least 2." >&2
  exit 2
fi
if [[ "${performance_mode}" == "turbo" ]] && (( steps < 5 || steps > 9 )); then
  printf '%s\n' "Turbo mode requires 5 to 9 scheduler points (4 to 8 model evaluations)." >&2
  exit 2
fi
if [[ ! "${seed}" =~ ^-?[0-9]+$ ]]; then
  printf '%s\n' "Seed must be an integer." >&2
  exit 2
fi
if [[ "${attention}" != "flash" && "${attention}" != "default" ]]; then
  printf '%s\n' "Attention must be either flash or default." >&2
  exit 2
fi
if [[ -z "${offload_margin}" ]]; then
  printf '%s\n' "Offload reserve margin must not be empty." >&2
  exit 2
fi
if [[ "${offload_mode}" != "copyback" && "${offload_mode}" != "cpu-master" && "${offload_mode}" != "pinned-master" ]]; then
  printf '%s\n' "Offload mode must be copyback, cpu-master, or pinned-master." >&2
  exit 2
fi
if [[ -z "${device}" ]]; then
  printf '%s\n' "Device must not be empty." >&2
  exit 2
fi
if [[ ! -d "${model_dir}" ]]; then
  printf 'Model directory does not exist: %s\n' "${model_dir}" >&2
  exit 2
fi

printf '%s\n' "Paste a multiline prompt. Enter ::end on its own line to generate:"
prompt_lines=()
while IFS= read -r line; do
  if [[ "${line}" == "::end" ]]; then
    break
  fi
  prompt_lines+=("${line}")
done

prompt="$(printf '%s\n' "${prompt_lines[@]}")"
if [[ -z "${prompt//[[:space:]]/}" ]]; then
  printf '%s\n' "Prompt must not be empty." >&2
  exit 2
fi

output_path="${ROOT_DIR}/outputs/${output_name}"
command=(
  "${RUNNER}"
  --model "${model_dir}"
  --workflow "${workflow}"
  --prompt "${prompt}"
  --width "${width}"
  --height "${height}"
  --num-frames "${num_frames}"
  --steps "${steps}"
  --seed "${seed}"
  --attention "${attention}"
  --offload-margin "${offload_margin}"
  --offload-mode "${offload_mode}"
  --device "${device}"
  --output "${output_path}"
)
if [[ -n "${image}" ]]; then
  command+=(--image "$(realpath "${image}")")
fi
if [[ -n "${last_image}" ]]; then
  command+=(--last-image "$(realpath "${last_image}")")
fi
if [[ "${performance_mode}" == "turbo" ]]; then
  command+=(--lora "${turbo_lora}" --lora-scale "${lora_scale}")
fi

printf 'Output file: %s\n' "${output_path}"
exec "${command[@]}" "$@"
