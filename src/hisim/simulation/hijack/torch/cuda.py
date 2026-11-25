"""Patch PyTorch CUDA and Triton for driverless simulation hosts.

Process-wide runtime patches run before any framework import. The module hook
scopes the simulated node's GPU count to consumers that size deployments by
the local device count; patching torch globally would make vLLM probe GPUs.
"""

import os

import torch

from hisim.hook import BaseHook


def _patch_triton_for_cpu():
    """Allow driverless Triton imports and preload its CUDA language helpers.

    vLLM permits zero active drivers only with an empty device mask, so clear
    masks assigned by launchers (e.g. DS_GPU_NUM) when no CUDA device exists.
    Real GPU hosts keep their mask; missing Triton helpers remain optional.
    """
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import triton.language.extra.cuda  # noqa: F401
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass


def _patch_torch_cuda_for_cpu():
    """Supply fallback CUDA properties and initialization on driverless hosts."""
    if torch.cuda.is_available():
        return

    class _FakeDeviceProperties:
        major = 9
        minor = 0
        name = "NVIDIA Simulated GPU"
        total_memory = 80 * (1 << 30)  # 80 GiB
        multi_processor_count = 132

    _original_get_device_properties = torch.cuda.get_device_properties

    def _mock_get_device_properties(device=None):
        try:
            return _original_get_device_properties(device)
        except (RuntimeError, AssertionError):
            return _FakeDeviceProperties()

    torch.cuda.get_device_properties = _mock_get_device_properties

    _original_lazy_init = torch.cuda._lazy_init

    def _mock_lazy_init():
        try:
            _original_lazy_init()
        except RuntimeError:
            pass

    torch.cuda._lazy_init = _mock_lazy_init


def patch_cuda_runtime():
    """Apply process-wide patches; call before importing inference frameworks."""
    _patch_triton_for_cpu()
    _patch_torch_cuda_for_cpu()


def _node_devices() -> int:
    from hisim.simulation.manager import ConfigManager
    from hisim.simulation.types import PlatformConfig

    platform = ConfigManager._get_raw_config().get("platform", {})
    devices = platform.get("num_device_per_node")
    return int(devices if devices is not None else PlatformConfig.num_device_per_node)


class _CudaView:
    def device_count(self) -> int:
        return _node_devices()

    def __getattr__(self, name):
        return getattr(torch.cuda, name)


class _TorchView:
    cuda = _CudaView()

    def __getattr__(self, name):
        return getattr(torch, name)


class M_TorchNodeDeviceCountHook(BaseHook):
    """Report `platform.num_device_per_node` as the local GPU count.

    Consumers treat a TP size above the local GPU count as multi-node TP; with
    no GPUs, every multi-node deployment (e.g. DP across nodes) would take it:
    - DashLLM's vLLM backend starts a headless executor on non-zero ranks.
    - The hybrid connector's workers send scheduler RPCs to MASTER_ADDRESS
      instead of their own EngineCore.
    Rebinding only the consumer modules' lookups simulates the count.
    """

    HOOK_MODULE_NAME = (
        r"^(dashllm\.core\.backend\._backend_vllm"
        r"|vllm\.v1\.hybrid_connector\.engine_proxy)$"
    )
    REGEX = True

    @classmethod
    def hook(cls, target):
        # Simulation masks devices; is_available() can still report the driver.
        if torch.cuda.device_count() > 0:
            return
        if target.__name__.startswith("dashllm."):
            target.torch = _TorchView()
        else:
            target.cuda_device_count_stateless = _node_devices
