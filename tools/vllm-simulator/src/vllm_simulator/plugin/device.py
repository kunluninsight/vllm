# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulated accelerator capacity and construction-time device properties."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulatorDevice:
    name: str
    total_memory_gb: float
    compute_capability: tuple[int, int]
    num_compute_units: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("simulator.device.name must be a non-empty string")
        if (
            type(self.total_memory_gb) not in (int, float)
            or not math.isfinite(self.total_memory_gb)
            or self.total_memory_gb <= 0
        ):
            raise ValueError(
                "simulator.device.total_memory_gb must be finite and positive"
            )
        if type(self.num_compute_units) is not int or self.num_compute_units <= 0:
            raise ValueError(
                "simulator.device.num_compute_units must be a positive integer"
            )
        capability = self.compute_capability
        if (
            not isinstance(capability, (list, tuple))
            or len(capability) != 2
            or any(type(part) is not int or part < 0 for part in capability)
            or capability[0] == 0
        ):
            raise ValueError(
                "simulator.device.compute_capability must be [major, minor]"
            )
        object.__setattr__(self, "compute_capability", tuple(capability))

    @property
    def total_memory_bytes(self) -> int:
        return int(self.total_memory_gb * 1_000_000_000)

    @classmethod
    def from_config(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("simulator.device must be a resolved device specification")
        return cls(**value)
