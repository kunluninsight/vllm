# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve native V1 cache layouts without weights or accelerator allocation."""

import sys
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
from functools import cache
from itertools import chain
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.overrides import TorchFunctionMode

from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.plugin.device import SimulatorDevice


@dataclass(frozen=True)
class ModelLayout:
    layer_configs: dict
    model_memory_bytes: int
    supported_kv_cache_layouts: tuple[str, ...]


def model_storage_bytes(model):
    """Count unique parameter and buffer storage, including packed quantization."""
    storages = {
        tensor.untyped_storage()._cdata: tensor.untyped_storage()
        for tensor in chain(model.parameters(), model.buffers())
    }
    return sum(storage.nbytes() for storage in storages.values())


class _ForceMetaDevice(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if kwargs.get("device") is not None:
            kwargs["device"] = "meta"
        return func(*args, **kwargs)


@dataclass
class _VirtualProcessGroup:
    world_size: int
    group_rank: int

    def size(self):
        return self.world_size

    def rank(self):
        return self.group_rank


class _NullCudaObject:
    def __init__(self, *args, **kwargs):
        pass


@contextmanager
def virtual_parallel_groups(parallel, pp_rank):
    """Lend model constructors rank metadata, restoring native groups on exit."""
    import vllm.distributed.parallel_state as ps

    sizes = {
        "_WORLD": parallel.world_size,
        "_TP": parallel.tensor_parallel_size,
        "_PP": parallel.pipeline_parallel_size,
        "_DP": parallel.data_parallel_size,
        "_PCP": parallel.prefill_context_parallel_size,
        "_DCP": parallel.decode_context_parallel_size,
        "_EP": parallel.tensor_parallel_size * parallel.data_parallel_size
        if parallel.enable_expert_parallel
        else 1,
        "_ETP": parallel.tensor_parallel_size,
    }
    with ExitStack() as stack:
        for name, size in sizes.items():
            group = object.__new__(ps.GroupCoordinator)
            group.world_size = size
            group.rank_in_group = pp_rank if name == "_PP" else 0
            group.rank = group.rank_in_group
            group.local_rank = 0
            group.ranks = list(range(size))
            group.device_group = _VirtualProcessGroup(size, group.rank_in_group)
            group.cpu_group = group.device_group
            group.device_communicator = None
            stack.enter_context(patch.object(ps, name, group))
        yield


@contextmanager
def meta_model_context(device: SimulatorDevice, *, model_type=None):
    """Select GPU layouts while redirecting even explicit CUDA tensors to meta."""
    import vllm.envs as envs
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.platforms import current_platform
    from vllm.platforms.cuda import CudaPlatform
    from vllm.v1.attention import selector
    from vllm.v1.attention.backends.mla.prefill import selector as prefill_selector
    from vllm.v1.attention.backends.mla.prefill.cpu_sdpa import CPUSDPAMLAPrefillBackend

    def layout_backend(cls, selected, attn_selector_config, num_heads=None):
        # Context parallelism and connectors affect execution support, not packing.
        selector = attn_selector_config._replace(
            use_pcp=False, use_dcp=False, use_kv_connector=False
        )
        return CudaPlatform.get_attn_backend_cls.__func__(
            cls, selected, selector, num_heads
        )

    capability = device.compute_capability
    platform_cls = type(current_platform)

    def clear_flashinfer_device_cache():
        module = sys.modules.get("flashinfer.utils")
        if module is not None:
            for name in (
                "get_device_properties",
                "get_device_sm_count",
                "get_compute_capability",
            ):
                getattr(module, name).cache_clear()

    with ExitStack() as stack:
        # Native selector keys omit the device; keep their memoization scoped.
        for module, name in (
            (selector, "_cached_get_attn_backend"),
            (prefill_selector, "_auto_select_mla_prefill_backend"),
        ):
            original = getattr(module, name)
            stack.enter_context(patch.object(module, name, cache(original.__wrapped__)))
        clear_flashinfer_device_cache()
        stack.callback(clear_flashinfer_device_cache)
        overrides = {
            "is_cuda": lambda self: True,
            "get_mla_prefill_backend_cls": classmethod(
                lambda cls, selector: CPUSDPAMLAPrefillBackend
            ),
            "get_cuda_runtime_major": classmethod(
                lambda cls: int((torch.version.cuda or "0").split(".")[0])
            ),
            "get_attn_backend_cls": classmethod(layout_backend),
            "get_valid_backends": classmethod(CudaPlatform.get_valid_backends.__func__),
        }
        for name, value in overrides.items():
            stack.enter_context(patch.object(platform_cls, name, value, create=True))
        properties = SimpleNamespace(
            major=capability[0],
            minor=capability[1],
            multi_processor_count=device.num_compute_units,
            name=device.name,
            total_memory=device.total_memory_bytes,
        )
        stack.enter_context(
            patch.object(
                torch.cuda, "get_device_properties", lambda device=None: properties
            )
        )
        stack.enter_context(
            patch.object(
                torch.cuda, "get_device_capability", lambda device=None: capability
            )
        )
        stack.enter_context(patch.object(torch.cuda, "Stream", _NullCudaObject))
        stack.enter_context(patch.object(torch.cuda, "Event", _NullCudaObject))
        # These paths select or warm up compute kernels, never cache layouts.
        for name in ("VLLM_KIMI_K3_GEMM_AR", "VLLM_ENABLE_GEMM_RS"):
            stack.enter_context(patch.object(envs, name, False))
        stack.enter_context(
            patch.object(sparse_attn_indexer, "has_deep_gemm", lambda: True)
        )
        if model_type == "kimi_linear":
            from vllm.models.kimi_k3.nvidia.latent_moe_runner import LatentMoERunner

            stack.enter_context(
                patch.object(LatentMoERunner, "_use_fused_path", lambda self: False)
            )
        from vllm.model_executor.layers.quantization import fp8

        stack.enter_context(
            patch.object(fp8, "init_fp8_linear_kernel", lambda **kwargs: None)
        )
        stack.enter_context(torch.device("meta"))
        stack.enter_context(_ForceMetaDevice())
        yield


def build_model_layout(config):
    """Resolve native layouts, budgeting for each cache kind's largest PP stage."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed.kv_transfer.kv_connector.utils import (
        get_current_attn_backends,
    )
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_layer_configs

    simulation = SimulatorConfig.from_config(config)
    view = simulation.predictor_config(config)
    # Preserve native packed weights when their footprint determines capacity.
    view.quant_config = deepcopy(config.quant_config)
    view.kernel_config = copy(config.kernel_config)
    view.kernel_config.enable_jit_warmup = False
    view.attention_config = copy(config.attention_config)
    view.attention_config.sparse_mla_shard_q = False
    view.model_config = copy(config.model_config)
    view.compilation_config = deepcopy(config.compilation_config)
    view.compilation_config.static_forward_context = {}
    stage_layers = []
    stage_memory = []
    with (
        set_current_vllm_config(view),
        meta_model_context(
            simulation.device,
            model_type=config.model_config.hf_text_config.model_type,
        ),
        set_default_torch_dtype(view.model_config.dtype),
    ):
        for rank in range(view.parallel_config.pipeline_parallel_size):
            before = set(view.compilation_config.static_forward_context)
            with virtual_parallel_groups(view.parallel_config, rank):
                model = initialize_model(view)
                stage_layers.append(
                    set(view.compilation_config.static_forward_context) - before
                )
                stage_memory.append(model_storage_bytes(model))
                del model
        with virtual_parallel_groups(view.parallel_config, 0):
            current_platform.update_block_size_for_backend(view)
            layers = get_kv_cache_layer_configs(view)
            supported_layouts = tuple(
                layout.name
                for layout in get_supported_kv_cache_layouts(
                    get_current_attn_backends(view)
                )
            )

    # A single scheduler represents PP. Budget for the busiest stage of each
    # cache kind, including stages with different attention/state mixtures.
    largest_groups = {}
    for names in stage_layers:
        groups = defaultdict(dict)
        for name, layer in layers.items():
            if name not in names:
                continue
            groups[layer][name] = layer
        for layer, group in groups.items():
            if len(group) > len(largest_groups.get(layer, {})):
                largest_groups[layer] = group
    result = {
        name: layer
        for group in largest_groups.values()
        for name, layer in group.items()
    }
    if not result:
        raise ValueError("The simulator model did not register any KV cache layers")
    dcp = view.parallel_config.decode_context_parallel_size
    if dcp > 1:
        result = {
            name: replace(
                layer,
                spec=replace(
                    layer.spec,
                    block_size=layer.spec.block_size * dcp,
                    tokens_per_state=layer.spec.tokens_per_state * dcp,
                ),
            )
            if layer.spec.dcp_sharded
            else layer
            for name, layer in result.items()
        }
    for field in fields(view.cache_config):
        setattr(config.cache_config, field.name, getattr(view.cache_config, field.name))
    config.cache_config.block_size = (
        view.cache_config.block_size * view.parallel_config.decode_context_parallel_size
    )
    config.additional_config["simulator"]["physical_kv_cache_block_size"] = (
        view.cache_config.block_size
    )
    return ModelLayout(result, max(stage_memory), supported_layouts)
