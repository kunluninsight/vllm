"""Hook providers for PyTorch and Triton on simulation hosts."""

from hisim.hook import BaseHook

__all__ = ("patch_runtime", "get_class_hooks", "get_module_hooks")


def patch_runtime() -> None:
    """Apply process-wide torch/Triton patches before framework imports."""
    from .cuda import patch_cuda_runtime

    patch_cuda_runtime()


def get_class_hooks() -> list[type[BaseHook]]:
    return []


def get_module_hooks() -> list[type[BaseHook]]:
    from .cuda import M_TorchNodeDeviceCountHook

    return [M_TorchNodeDeviceCountHook]
