"""Install simulator hooks before importing the inference frameworks."""

import logging
import os
import re
import sys

import torch


def _patch_triton_for_cpu():
    """Allow driverless Triton imports and preload its CUDA language helpers.

    vLLM permits zero active drivers with an empty device mask. Preserve an
    existing mask; missing Triton helpers remain optional.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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


# Explicit entrypoints and the .pth bootstrap can share the same interpreter.
_HOOKS_INSTALLED = False


def init_hook():
    """Install hooks once before framework imports; the bootstrap gates activation."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return True

    # The framework-owned controls stay in startup, not simulator env.py.
    os.environ.setdefault("SRPC_STREAM_DISABLE_RDMA", "1")
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "42"
        logging.getLogger("hisim").warning(
            "[init_hook] PYTHONHASHSEED not set, defaulting to 42 for cross-node hash consistency"
        )

    _patch_triton_for_cpu()
    _patch_torch_cuda_for_cpu()

    import hisim.hook as hisim_hook
    from hisim.simulation.hijack import dashllm, v6d, vllm

    class_hooks = [
        *vllm.get_class_hooks(),
        *dashllm.get_class_hooks(),
        *v6d.get_class_hooks(),
    ]
    module_hooks = [
        *vllm.get_module_hooks(),
        *dashllm.get_module_hooks(),
        *v6d.get_module_hooks(),
    ]
    class_hooks = list(dict.fromkeys(class_hooks))
    module_hooks = list(dict.fromkeys(module_hooks))
    if class_hooks:
        hisim_hook.install_class_hooks(class_hooks)
    if module_hooks:
        hisim_hook.install_module_hooks(module_hooks)

    # Patch loaded modules now; the finder handles future imports.
    logger = logging.getLogger("hisim")
    for hook in module_hooks:
        for name, module in tuple(sys.modules.items()):
            if module is None:
                continue
            matches = (
                bool(re.search(hook.HOOK_MODULE_NAME, name))
                if hook.REGEX
                else name == hook.HOOK_MODULE_NAME
            )
            if matches:
                try:
                    hook.hook(module)
                except Exception as exc:
                    logger.warning("Failed to hook module [%s]. Error: %s", name, exc)

    _HOOKS_INSTALLED = True
    logger.info(
        "[init_hook] Installed %d class hooks and %d module hooks",
        len(class_hooks),
        len(module_hooks),
    )
    return True
