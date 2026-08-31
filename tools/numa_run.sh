#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${MINIMAX_H3_RUNNER:-${ROOT_DIR}/run.sh}"

device="mlu:0"
args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
  if [[ "${args[index]}" == "--device" ]] && (( index + 1 < ${#args[@]} )); then
    device="${args[index + 1]}"
  elif [[ "${args[index]}" == --device=* ]]; then
    device="${args[index]#--device=}"
  fi
done

if [[ ! "${device}" =~ ^mlu:([0-9]+)$ ]]; then
  printf 'NUMA launcher only supports an explicit MLU device, got %s.\n' "${device}" >&2
  exit 2
fi
logical_index="${BASH_REMATCH[1]}"
physical_index="${logical_index}"
if [[ -n "${MLU_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_devices <<<"${MLU_VISIBLE_DEVICES}"
  if (( logical_index < ${#visible_devices[@]} )) && [[ "${visible_devices[logical_index]}" =~ ^[0-9]+$ ]]; then
    physical_index="${visible_devices[logical_index]}"
  fi
fi

if ! command -v numactl >/dev/null 2>&1 || ! command -v cnmon >/dev/null 2>&1; then
  printf '%s\n' "numactl or cnmon is unavailable; running without NUMA binding." >&2
  exec env MINIMAX_H3_NUMA_BOUND=1 "${RUNNER}" "${args[@]}"
fi

pci_fields="$(
  cnmon info -c "${physical_index}" 2>/dev/null | awk '
    /^[[:space:]]+PCI[[:space:]]*$/ { in_pci = 1; next }
    in_pci && /Domain ID/ { domain = $NF }
    in_pci && /Bus num/ { bus = $NF }
    in_pci && /^[[:space:]]+Device[[:space:]]*:/ { device = $NF }
    in_pci && /Function/ { function_id = $NF }
    END { if (domain != "" && bus != "" && device != "" && function_id != "") print domain, bus, device, function_id }
  '
)"
read -r domain bus pci_device function <<<"${pci_fields}"

if [[ -z "${domain:-}" || -z "${bus:-}" || -z "${pci_device:-}" || -z "${function:-}" ]]; then
  printf '%s\n' "Could not resolve the MLU PCI address; running without NUMA binding." >&2
  exec env MINIMAX_H3_NUMA_BOUND=1 "${RUNNER}" "${args[@]}"
fi

pci_address="${domain}:${bus}:${pci_device}.${function}"
numa_file="/sys/bus/pci/devices/${pci_address}/numa_node"
if [[ ! -r "${numa_file}" ]]; then
  printf 'No NUMA metadata for MLU %s at %s; running without binding.\n' "${logical_index}" "${pci_address}" >&2
  exec env MINIMAX_H3_NUMA_BOUND=1 "${RUNNER}" "${args[@]}"
fi

numa_node="$(<"${numa_file}")"
if [[ ! "${numa_node}" =~ ^[0-9]+$ ]]; then
  printf 'MLU %s reports no local NUMA node; running without binding.\n' "${logical_index}" >&2
  exec env MINIMAX_H3_NUMA_BOUND=1 "${RUNNER}" "${args[@]}"
fi

printf 'Binding MLU %s (%s) to NUMA node %s.\n' "${logical_index}" "${pci_address}" "${numa_node}" >&2
exec numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
  env MINIMAX_H3_NUMA_BOUND=1 "${RUNNER}" "${args[@]}"
