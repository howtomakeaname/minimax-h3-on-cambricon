from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from types import MethodType

import torch


logger = logging.getLogger(__name__)

OFFLOAD_MODES = ("copyback", "cpu-master", "pinned-master")
DEFAULT_PINNED_COMPONENTS = ("text_encoder", "transformer", "transformer_ref")


@dataclass(frozen=True)
class CpuMasterStats:
    components: int = 0
    tensors: int = 0
    bytes: int = 0
    pinned_bytes: int = 0

    def __add__(self, other: "CpuMasterStats") -> "CpuMasterStats":
        return CpuMasterStats(
            components=self.components + other.components,
            tensors=self.tensors + other.tensors,
            bytes=self.bytes + other.bytes,
            pinned_bytes=self.pinned_bytes + other.pinned_bytes,
        )


def _component_base_name(component_id: str) -> str:
    return component_id.rsplit("_", 1)[0]


def _resolve_parent(module: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    path, _, leaf = name.rpartition(".")
    return (module.get_submodule(path) if path else module), leaf


def _set_parameter_data(module: torch.nn.Module, name: str, data: torch.Tensor) -> None:
    parent, leaf = _resolve_parent(module, name)
    parent._parameters[leaf].data = data


def _module_device(module: torch.nn.Module) -> torch.device:
    first = next(module.parameters(), None)
    if first is None:
        first = next(module.buffers(), None)
    return first.device if first is not None else torch.device("cpu")


def _capture_cpu_master(hook, module: torch.nn.Module, *, pin_memory: bool) -> CpuMasterStats:
    tensors = []
    total_bytes = 0
    pinned_bytes = 0
    for name, tensor in module.named_parameters(remove_duplicate=True):
        if tensor.device.type != "cpu":
            raise RuntimeError(f"CPU-master offload requires CPU weights, got {name} on {tensor.device}")
        cpu_data = tensor.detach()
        if pin_memory and cpu_data.numel() and not cpu_data.is_pinned():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The argument 'device' of Tensor.pin_memory")
                cpu_data = cpu_data.pin_memory(device="mlu")
            _set_parameter_data(module, name, cpu_data)
        parent, leaf = _resolve_parent(module, name)
        registered = parent._parameters[leaf]
        tensors.append((name, cpu_data, registered._version))
        size = cpu_data.numel() * cpu_data.element_size()
        total_bytes += size
        if cpu_data.is_pinned():
            pinned_bytes += size
    hook._cpu_master_tensors = tensors
    return CpuMasterStats(components=1, tensors=len(tensors), bytes=total_bytes, pinned_bytes=pinned_bytes)


def _restore_cpu_master(hook, module: torch.nn.Module) -> bool:
    if getattr(hook, "_cpu_master_disabled", False):
        return False
    for name, _, version in hook._cpu_master_tensors:
        parent, leaf = _resolve_parent(module, name)
        tensor = parent._parameters[leaf]
        if tensor._version != version:
            return False

    # Buffers may be legitimate mutable model state, so preserve their current values with the normal D2H path.
    # They are tiny relative to H3's parameters. Preserve aliases when the same buffer is registered more than once.
    moved_buffers: dict[int, torch.Tensor] = {}
    for submodule in module.modules():
        for name, buffer in submodule._buffers.items():
            if buffer is None or buffer.device.type == "cpu":
                continue
            buffer_id = id(buffer)
            if buffer_id not in moved_buffers:
                moved_buffers[buffer_id] = buffer.to("cpu")
            submodule._buffers[name] = moved_buffers[buffer_id]

    for name, cpu_data, _ in hook._cpu_master_tensors:
        _set_parameter_data(module, name, cpu_data)
    return True


def enable_cpu_master_offload(
    manager,
    *,
    mode: str,
    pinned_components: Iterable[str] = DEFAULT_PINNED_COMPONENTS,
) -> CpuMasterStats:
    """Convert Diffusers auto-offload hooks to discard accelerator copies of read-only weights.

    Call this immediately after ``enable_auto_cpu_offload`` and after applying all weight mutations such as LoRA.
    ``pinned-master`` pins only the selected component names; all other components keep pageable CPU masters.
    """
    if mode not in OFFLOAD_MODES:
        raise ValueError(f"Unsupported offload mode {mode!r}; expected one of {OFFLOAD_MODES}")
    if mode == "copyback":
        return CpuMasterStats()
    if not manager.model_hooks:
        raise RuntimeError("Automatic CPU offload must be enabled before CPU-master offload")

    pinned = set(pinned_components)
    total = CpuMasterStats()
    for user_hook in manager.model_hooks:
        hook = user_hook.hook
        module = user_hook.model
        component_name = _component_base_name(user_hook.model_id)
        if not isinstance(module, torch.nn.Module):
            continue
        if hasattr(hook, "_cpu_master_tensors"):
            raise RuntimeError(f"CPU-master offload is already enabled for {component_name}")

        should_pin = mode == "pinned-master" and component_name in pinned
        stats = _capture_cpu_master(hook, module, pin_memory=should_pin)
        original_init_hook = hook.init_hook

        def restore_or_copyback(self, target, *, _original=original_init_hook, _name=component_name):
            if _module_device(target).type == "cpu":
                return target
            if _restore_cpu_master(self, target):
                return target
            logger.warning("Weights changed for %s; falling back to copyback offload", _name)
            self._cpu_master_disabled = True
            self._cpu_master_tensors.clear()
            return _original(target)

        hook.init_hook = MethodType(restore_or_copyback, hook)
        total += stats
    return total
