# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute native CPU offload capacity from the original parallel topology."""

from dataclasses import replace

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm_simulator.plugin.config import SimulatorConfig


def cpu_offloading_spec(config, kv_cache_config) -> CPUOffloadingSpec:
    """Size the native CPU chunks using the original worker count and alignment."""
    native = build_offloading_config(config, kv_cache_config)
    original = SimulatorConfig.from_config(config).original_parallel_config
    native = replace(
        native, parallel=replace(native.parallel, world_size=original.world_size)
    )
    return CPUOffloadingSpec(native)
