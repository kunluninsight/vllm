"""Discover native KV metadata on meta tensors, independent of execution ranks."""

import copy
import ctypes
import os
import sys
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from hisim.simulation.manager import ConfigManager
from hisim.utils import get_logger

logger = get_logger()
_LAYOUT_LOCK = threading.RLock()
_ALIGNED_CACHE_FIELDS = ("block_size", "mamba_block_size", "mamba_page_size_padded")


class _ForceMetaFunctions(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if "device" in kwargs:
            kwargs["device"] = torch.device("meta")
        return func(*args, **kwargs)


class _ForceMetaDevice(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.name() == "_C::get_cuda_view_from_cpu_tensor" and args[0].is_meta:
            # PLE creates a UVA alias during layout discovery, where no storage exists.
            return args[0].detach()
        kwargs = dict(kwargs or {})
        if "device" in kwargs:
            kwargs["device"] = torch.device("meta")
        return func(*args, **kwargs)


class _NullCudaStream:
    cuda_stream = 0
    device = torch.device("meta")

    def __init__(self, *args, **kwargs):
        pass

    def wait_stream(self, stream):
        pass

    def wait_event(self, event):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True

    def record_event(self, event=None):
        return event or _NullCudaEvent()


class _NullCudaEvent:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.0


def _capability_number(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]) * 10 + int(value[1])
    text = str(value)
    if "." in text:
        major, minor = text.split(".", 1)
        return int(major) * 10 + int(minor)
    return int(value)


def _device_capability():
    from vllm.platforms.interface import DeviceCapability

    accelerator = (
        ConfigManager._get_raw_config().get("platform", {}).get("accelerator", {})
    )
    name = accelerator.get("name", "h100_sxm").lower()
    explicit = accelerator.get("compute_capability")
    if explicit is not None:
        value = _capability_number(explicit)
        return DeviceCapability(value // 10, value % 10)
    if any(
        key in name
        for key in ("b200", "b100", "gb200", "l20n", "l20a", "l20b", "l20c", "l20d")
    ):
        return DeviceCapability(10, 0)
    if any(key in name for key in ("h100", "h200", "h20")):
        return DeviceCapability(9, 0)
    if any(key in name for key in ("l20", "l40", "l4", "4090")):
        return DeviceCapability(8, 9)
    if "a100" in name:
        return DeviceCapability(8, 0)
    raise ValueError(
        f"KV meta layout needs compute_capability for accelerator {name!r}"
    )


def _cuda_platform_base():
    try:
        from vllm.platforms.cuda import CudaPlatformBase
    except ImportError as e:
        # vllm.platforms.cuda imports vllm._C, whose kernels meta constructors bind;
        # CPU hosts fail with "libcuda.so.1: cannot open shared object file".
        # The toolkit's driver stub satisfies linking; CUDA stays unavailable.
        if "libcuda" not in str(e):
            raise
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        stubs = [
            os.path.join(cuda_home, sub, "libcuda.so")
            for sub in ("lib64/stubs", "targets/x86_64-linux/lib/stubs")
        ]
        stub = next((path for path in stubs if os.path.exists(path)), None)
        if stub is None:
            raise ImportError(f"{e}; no CUDA driver stub in {stubs}") from e
        ctypes.CDLL(stub, mode=ctypes.RTLD_GLOBAL)
        logger.info("[VLLM Hijack] Loaded CUDA driver stub %s for vllm._C", stub)
        from vllm.platforms.cuda import CudaPlatformBase
    return CudaPlatformBase


@contextmanager
def _cuda_metadata_platform():
    """Limit CUDA-shaped constructor probes to this startup-only scope."""
    from vllm.platforms import current_platform

    CudaPlatformBase = _cuda_platform_base()
    capability = _device_capability()
    platform_cls = type(current_platform)

    class _MetadataCudaSelector(CudaPlatformBase):
        @classmethod
        def get_device_capability(cls, device_id=0):
            return capability

    attrs = {
        "is_cuda": lambda self: True,
        "is_cuda_alike": lambda self: True,
        "get_device_capability": classmethod(lambda cls, device_id=0: capability),
        "has_device_capability": classmethod(
            lambda cls, required, device_id=0: (
                capability.to_int() >= _capability_number(required)
            )
        ),
        "is_device_capability": classmethod(
            lambda cls, required, device_id=0: (
                capability.to_int() == _capability_number(required)
            )
        ),
        "get_cuda_runtime_major": classmethod(
            lambda cls: int((torch.version.cuda or "0").split(".")[0])
        ),
        "num_compute_units": classmethod(lambda cls, device_id=0: 132),
        "get_attn_backend_cls": classmethod(
            lambda cls, *args, **kwargs: _MetadataCudaSelector.get_attn_backend_cls(
                *args, **kwargs
            )
        ),
    }
    with ExitStack() as stack:
        for name, attr in attrs.items():
            stack.enter_context(patch.object(platform_cls, name, attr, create=True))
        # Only MoE constructor selection needs OOT; no expert execution takes place.
        stack.enter_context(
            patch.object(current_platform, "is_out_of_tree", lambda: True)
        )
        stack.enter_context(patch.object(torch.cuda, "Stream", _NullCudaStream))
        stack.enter_context(patch.object(torch.cuda, "Event", _NullCudaEvent))
        # Hyperconnection constructors probe current_device(); otherwise they raise
        # "CUDA error: no CUDA-capable device is detected" before meta redirection.
        stack.enter_context(patch.object(torch.cuda, "current_device", lambda: 0))
        properties = SimpleNamespace(
            major=capability.major,
            minor=capability.minor,
            name="HiSim metadata device",
            total_memory=80 * (1 << 30),
            multi_processor_count=132,
        )
        stack.enter_context(
            patch.object(
                torch.cuda, "get_device_properties", lambda *a, **k: properties
            )
        )
        stack.enter_context(
            patch.object(
                torch.cuda,
                "get_device_capability",
                lambda *a, **k: (capability.major, capability.minor),
            )
        )
        stack.enter_context(
            patch.object(
                torch.cuda, "current_stream", lambda *a, **k: _NullCudaStream()
            )
        )
        stack.enter_context(
            patch.object(
                torch.cuda, "default_stream", lambda *a, **k: _NullCudaStream()
            )
        )
        stack.enter_context(
            patch.object(torch.cuda, "stream", lambda stream: nullcontext())
        )
        yield current_platform


def target_layout_config(vllm_config):
    """A shallow config view for layout-only native calls, never for RPC execution."""
    inputs = getattr(vllm_config, "_hisim_layout_inputs", {})
    tp = int(inputs.get("tp_size", 1))
    pp = int(inputs.get("pp_size", 1))
    if tp < 1:
        raise ValueError("Target TP must be positive")
    if pp < 1:
        raise ValueError("Target PP must be positive")
    layout = copy.copy(vllm_config)
    # Native MTP copies the layout and expects only declared VllmConfig fields.
    if hasattr(layout, "_hisim_layout_inputs"):
        del layout._hisim_layout_inputs
    layout.parallel_config = copy.copy(vllm_config.parallel_config)
    layout.parallel_config.tensor_parallel_size = tp
    layout.parallel_config.pipeline_parallel_size = pp
    if hasattr(layout.parallel_config, "prefill_context_parallel_size"):
        layout.parallel_config.prefill_context_parallel_size = max(
            1, int(inputs.get("pcp_size", 1))
        )
    if hasattr(layout.parallel_config, "decode_context_parallel_size"):
        layout.parallel_config.decode_context_parallel_size = max(
            1, int(inputs.get("dcp_size", 1))
        )
    # local_world_size is derived from this stored field in PAI-vLLM.
    layout.parallel_config.world_size = tp * pp
    layout.cache_config = copy.deepcopy(vllm_config.cache_config)
    layout.quant_config = copy.deepcopy(vllm_config.quant_config)
    return layout


def align_target_cache(vllm_config):
    """Recompute native alignment from user input, before scheduler/hash creation."""
    from vllm.model_executor.models.config import HybridAttentionMambaModelConfig

    layout = target_layout_config(vllm_config)
    inputs = getattr(vllm_config, "_hisim_layout_inputs", {})
    scheduler = ConfigManager._get_raw_config().get("scheduler", {})
    requested = scheduler.get("physical_block_size", inputs.get("block_size"))
    layout.cache_config.block_size = requested
    layout.cache_config.mamba_block_size = inputs.get("mamba_block_size")
    layout.cache_config.mamba_page_size_padded = inputs.get("mamba_page_size_padded")
    if layout.model_config.is_hybrid:
        with _LAYOUT_LOCK, _cuda_metadata_platform():
            HybridAttentionMambaModelConfig.verify_and_update_config(layout)
    elif requested is None:
        layout.cache_config.block_size = vllm_config.cache_config.block_size
    if (
        scheduler.get("physical_block_size") is not None
        and layout.cache_config.block_size != requested
    ):
        raise ValueError(
            f"physical_block_size={requested} is incompatible with native target layout; "
            f"minimum/aligned size is {layout.cache_config.block_size}"
        )
    for field in _ALIGNED_CACHE_FIELDS:
        setattr(vllm_config.cache_config, field, getattr(layout.cache_config, field))
    return layout


def build_kv_cache_metadata(vllm_config, *, negotiate_layouts=False):
    """Get native layer specs and supported layouts from the meta model."""
    import vllm.distributed.parallel_state as ps
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    layout = target_layout_config(vllm_config)
    layout.compilation_config = copy.copy(vllm_config.compilation_config)
    layout.compilation_config.static_forward_context = {}
    tp_group = ps.get_tp_group()
    with _LAYOUT_LOCK, ExitStack() as stack:
        stack.enter_context(
            patch.object(
                tp_group, "world_size", layout.parallel_config.tensor_parallel_size
            )
        )
        if getattr(ps, "_EP", None) is None:
            stack.enter_context(patch.object(ps, "_EP", tp_group, create=True))
        platform = stack.enter_context(_cuda_metadata_platform())
        stack.enter_context(set_current_vllm_config(layout))
        stack.enter_context(set_default_torch_dtype(layout.model_config.dtype))
        stack.enter_context(torch.device("meta"))
        stack.enter_context(_ForceMetaFunctions())
        stack.enter_context(_ForceMetaDevice())
        # Qwen4 constructors autotune decode kernels even on meta tensors;
        # otherwise Triton raises "0 active drivers ([]). There should only be one."
        qwen_next = sys.modules.get("vllm.model_executor.models.qwen3_next")
        if qwen_next is not None and hasattr(
            qwen_next, "warmup_fused_recurrent_autotune"
        ):
            stack.enter_context(
                patch.object(
                    qwen_next, "warmup_fused_recurrent_autotune", lambda *a, **k: None
                )
            )
        try:
            model = initialize_model(layout)
            draft = None
            speculative = layout.speculative_config
            if (
                speculative is not None
                and getattr(speculative, "draft_model_config", None) is not None
            ):
                draft = initialize_model(
                    layout, model_config=speculative.draft_model_config
                )
            # Recent runtimes postpone hybrid alignment until layers are available.
            update_block_size = getattr(
                type(platform), "update_block_size_for_backend", None
            )
            if negotiate_layouts and update_block_size is not None:
                update_block_size(layout)
                configured = ConfigManager._get_raw_config().get("scheduler", {})
                expected = configured.get("physical_block_size")
                if expected is not None and layout.cache_config.block_size != expected:
                    raise ValueError(
                        f"physical_block_size={expected} does not match native "
                        f"aligned block size {layout.cache_config.block_size}"
                    )
                for field in _ALIGNED_CACHE_FIELDS:
                    setattr(
                        vllm_config.cache_config,
                        field,
                        getattr(layout.cache_config, field),
                    )
            specs = get_kv_cache_spec(layout)
            layouts = []
            if negotiate_layouts:
                from vllm.distributed.kv_transfer.kv_connector.utils import (
                    get_current_attn_backends,
                )
                from vllm.v1.attention.backends.utils import (
                    get_supported_kv_cache_layouts,
                )

                backends = (
                    []
                    if layout.is_mm_encoder_only
                    else get_current_attn_backends(layout)
                )
                layouts = [
                    item.name for item in get_supported_kv_cache_layouts(backends)
                ]
            # The active v1 runner also preserves backend layout in attention specs.
            from vllm.v1.kv_cache_interface import AttentionSpec

            for name, spec in list(specs.items()):
                layer = layout.compilation_config.static_forward_context[name]
                if isinstance(spec, AttentionSpec):
                    backend = layer.get_attn_backend()
                    if hasattr(backend, "get_effective_kv_cache_layout"):
                        specs[name] = replace(
                            spec,
                            kv_cache_layout=backend.get_effective_kv_cache_layout(),
                        )
            del model, draft
        finally:
            layout.compilation_config.static_forward_context.clear()
    if not specs:
        raise RuntimeError("Native meta model produced no KV cache specs")
    logger.info(
        "[VLLM Hijack] Built KV cache spec via meta-device model walk: %d layers "
        "(target tp=%d, block_size=%d, mamba_page_size_padded=%s, layouts=%s)",
        len(specs),
        layout.parallel_config.tensor_parallel_size,
        vllm_config.cache_config.block_size,
        vllm_config.cache_config.mamba_page_size_padded,
        layouts,
    )
    return specs, layouts


def layout_backend_view(backend):
    """Reuse native byte accounting with target ranks and the actual KV descriptors."""
    return SimpleNamespace(
        vllm_config=target_layout_config(backend.vllm_config),
        kv_cache_config=backend.kv_cache_config,
        _enable_kvhead_dedup=backend._enable_kvhead_dedup,
    )
