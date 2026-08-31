from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from minimax_h3_mlu.offload import _restore_cpu_master, enable_cpu_master_offload


class FakeHook:
    def __init__(self):
        self.copyback_calls = 0

    def init_hook(self, module):
        self.copyback_calls += 1
        return module.to("cpu")


class FakeManager:
    def __init__(self, model):
        self.hook = FakeHook()
        self.model_hooks = [
            SimpleNamespace(
                model_id=f"transformer_{id(model)}",
                model=model,
                hook=self.hook,
            )
        ]


def test_copyback_mode_keeps_original_hook():
    model = torch.nn.Linear(4, 4)
    manager = FakeManager(model)

    stats = enable_cpu_master_offload(manager, mode="copyback")

    assert stats.bytes == 0
    assert not hasattr(manager.hook, "_cpu_master_tensors")


def test_cpu_master_restores_original_parameter_storage():
    model = torch.nn.Linear(4, 4)
    manager = FakeManager(model)
    original_weight = model.weight.detach()
    original_bias = model.bias.detach()

    stats = enable_cpu_master_offload(manager, mode="cpu-master")
    model.weight.data = model.weight.detach().clone()
    model.bias.data = model.bias.detach().clone()
    assert _restore_cpu_master(manager.hook, model) is True

    assert stats.components == 1
    assert stats.tensors == 2
    assert model.weight.data_ptr() == original_weight.data_ptr()
    assert model.bias.data_ptr() == original_bias.data_ptr()
    assert manager.hook.copyback_calls == 0


def test_mutated_weight_rejects_master_restore():
    model = torch.nn.Linear(4, 4)
    manager = FakeManager(model)
    enable_cpu_master_offload(manager, mode="cpu-master")

    with torch.no_grad():
        model.weight.add_(1)
    assert _restore_cpu_master(manager.hook, model) is False
    assert manager.hook.copyback_calls == 0


def test_enabling_twice_is_rejected():
    model = torch.nn.Linear(4, 4)
    manager = FakeManager(model)
    enable_cpu_master_offload(manager, mode="cpu-master")

    with pytest.raises(RuntimeError, match="already enabled"):
        enable_cpu_master_offload(manager, mode="cpu-master")


def test_invalid_mode_is_rejected():
    model = torch.nn.Linear(4, 4)
    manager = FakeManager(model)

    with pytest.raises(ValueError, match="Unsupported offload mode"):
        enable_cpu_master_offload(manager, mode="invalid")
