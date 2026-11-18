# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve native cache layouts into transport-safe metadata."""

import hashlib
import json
from dataclasses import dataclass, replace

from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.simple_kv_offload.sizing import gpu_total_bytes
from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.plugin.kv_transfer.protocol import WireKVLayout, head_range


def signature(value):
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class PDKVLayout:
    instance: str
    tp_size: int
    num_kv_heads: int
    block_size: int
    block_bytes: int
    cache_layout: str | None
    specs: tuple[tuple[str, AttentionSpec], ...]

    model_signature: str = ""

    def head_range(self, rank):
        return head_range(self.tp_size, self.num_kv_heads, rank)

    def canonical_specs(self):
        self.head_range(0)
        canonical = []
        for name, spec in self.specs:
            if not isinstance(spec, AttentionSpec) or (
                spec.tokens_per_state != 1
                or spec.num_heads != spec.num_kv_heads
                or spec.num_kv_heads != max(1, self.num_kv_heads // self.tp_size)
            ):
                raise ValueError("Heterogeneous TP requires dense per-head KV layouts")
            canonical.append(
                (
                    name,
                    replace(
                        spec,
                        num_kv_heads=self.num_kv_heads,
                        num_head_slots=None,
                        max_tp_shards=self.num_kv_heads,
                        page_size_padded=None,
                        block_stride_alignment=None,
                        kv_head_major_slots=False,
                    ),
                )
            )
        return tuple(canonical)

    def to_wire(self):
        try:
            canonical = signature(repr(self.canonical_specs()))
        except ValueError:
            canonical = None
        head_bytes = sum(
            spec.unpadded_page_size_bytes // spec.num_kv_heads for _, spec in self.specs
        )
        return WireKVLayout(
            self.instance,
            self.tp_size,
            self.num_kv_heads,
            self.block_size,
            self.block_bytes,
            self.cache_layout,
            signature(repr(self.specs)),
            canonical,
            head_bytes,
            self.model_signature,
        )

    def plan_to(self, destination):
        return self.to_wire().plan_to(destination.to_wire())

    @classmethod
    def from_config(cls, config, cache, instance):
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

        if len(cache.kv_cache_groups) != 1 or (
            cache.kv_cache_groups[0].manager_config.manager_cls
            is not FullAttentionManager
        ):
            raise ValueError("Simulator PD requires one full-attention KV group")
        group = cache.kv_cache_groups[0]
        original = SimulatorConfig.from_config(config).original_parallel_config
        model = config.model_config
        values = model.hf_config.to_dict()
        for key in ("_name_or_path", "_commit_hash", "transformers_version"):
            values.pop(key, None)
        identity = signature(json.dumps([values, str(model.dtype)], sort_keys=True))
        return cls(
            instance,
            original.tensor_parallel_size,
            1 if model.use_mla else model.get_total_num_kv_heads(),
            group.block_size,
            gpu_total_bytes(cache) // cache.num_blocks,
            cache.kv_cache_layout,
            tuple(sorted(group.kv_cache_specs.items())),
            identity,
        )
