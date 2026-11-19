"""KVCM daemon bootstrap must replace payload I/O, including earlier imports."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from sglang_simulator.simulation.vllm.v6d import v6d_capacity
from sglang_simulator.simulation.vllm.v6d.tair_kvcm import (
    C_TairKVCMTransferRuntimeHook,
)


def test_bootstrap_patches_preimported_transfer_runtime(monkeypatch):
    module = ModuleType(C_TairKVCMTransferRuntimeHook.HOOK_MODULE_NAME)

    class Runtime:
        def _create_transfer_client(self):
            raise AssertionError("Real payload client must be replaced in simulation")

    module.TairKVCMTransferRuntime = Runtime
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(v6d_capacity, "_RUNTIME_HOOKS_INSTALLED", False)
    register = Mock()
    monkeypatch.setattr(v6d_capacity, "install_class_hooks", register)

    v6d_capacity.install_v6d_runtime_hooks()
    Runtime()._create_transfer_client().close()
    first = Runtime._create_transfer_client
    v6d_capacity.install_v6d_runtime_hooks()
    C_TairKVCMTransferRuntimeHook.hook(Runtime)
    assert Runtime._create_transfer_client is first
    register.assert_called_once()


def test_bootstrap_keeps_future_import_hook(monkeypatch):
    monkeypatch.delitem(
        sys.modules, C_TairKVCMTransferRuntimeHook.HOOK_MODULE_NAME, raising=False
    )
    monkeypatch.setattr(v6d_capacity, "_RUNTIME_HOOKS_INSTALLED", False)
    register = Mock()
    monkeypatch.setattr(v6d_capacity, "install_class_hooks", register)
    v6d_capacity.install_v6d_runtime_hooks()
    assert C_TairKVCMTransferRuntimeHook in register.call_args.args[0]


def test_payload_replacement_keeps_batch_results_and_validation(monkeypatch):
    module = ModuleType(C_TairKVCMTransferRuntimeHook.HOOK_MODULE_NAME)
    recorded = []
    module._record_transfer_result = lambda *result: recorded.append(result)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Runtime:
        pass

    C_TairKVCMTransferRuntimeHook.hook(Runtime)
    runtime = Runtime()
    assert runtime.batch_load_to_memory([], [], []) is True
    assert runtime.batch_save_from_memory([], [], []) == []
    assert recorded == []
    assert runtime.batch_load_to_memory(["a", "b"], [1, 2], [3, 4]) is True
    assert runtime.batch_save_from_memory(["a", "b"], [1, 2], [3, 4]) == [True, True]
    assert recorded == [("read", True), ("write", True)]
    for method in (runtime.batch_load_to_memory, runtime.batch_save_from_memory):
        with pytest.raises(ValueError, match="length mismatch"):
            method(["a"], [], [3])
    assert recorded == [("read", True), ("write", True)]
