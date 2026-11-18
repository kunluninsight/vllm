# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulator configuration and preserved native parallel topology."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm_simulator.plugin.device import SimulatorDevice

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


@dataclass(frozen=True)
class SimulatorConfig:
    device: SimulatorDevice | dict | None = None
    token_id: int = 1
    mode: str = "NON_BLOCKING"
    predictor: dict = field(default_factory=dict)
    request_dispatcher: dict = field(default_factory=dict)
    step_trace_dir: str | None = None
    original_parallel_config: ParallelConfig | None = None
    physical_kv_cache_block_size: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "device", SimulatorDevice.from_config(self.device))
        if self.mode not in ("BLOCKING", "NON_BLOCKING"):
            raise ValueError("simulator.mode must be BLOCKING or NON_BLOCKING")
        if type(self.token_id) is not int or self.token_id < 0:
            raise ValueError("simulator.token_id must be a non-negative integer")
        for name in ("predictor", "request_dispatcher"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"simulator.{name} must be a JSON object")
        if self.original_parallel_config is not None:
            from vllm.config import ParallelConfig

            if not isinstance(self.original_parallel_config, ParallelConfig):
                raise TypeError(
                    "simulator.original_parallel_config is an internal backup"
                )

    @classmethod
    def from_config(cls, config):
        return cls(**config.additional_config.get("simulator", {}))

    @classmethod
    def backup_parallel_config(cls, config):
        """Capture requested parallelism and resolve the physical KV block size."""
        for name in ("request_dispatcher", "predictor"):
            if name in config.additional_config:
                raise ValueError(
                    f"Move additional_config.{name} to "
                    f"additional_config.simulator.{name}"
                )
        options = dict(config.additional_config.get("simulator", {}))
        if options.get("original_parallel_config") is None:
            options["original_parallel_config"] = deepcopy(config.parallel_config)
            options["physical_kv_cache_block_size"] = config.cache_config.block_size
        config.additional_config = {**config.additional_config, "simulator": options}
        return cls.from_config(config)

    def predictor_config(self, config):
        """Return a full config view with the requested simulation topology."""
        result = copy(config)
        result.parallel_config = deepcopy(
            self.original_parallel_config or config.parallel_config
        )
        for name in (
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_index",
        ):
            setattr(result.parallel_config, name, getattr(config.parallel_config, name))
        result.cache_config = copy(config.cache_config)
        if self.physical_kv_cache_block_size is not None:
            result.cache_config.block_size = self.physical_kv_cache_block_size
        return result
