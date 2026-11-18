# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical workload timing, independent of wall-clock request pacing."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    request_rate: float = float("inf")
    ignore_request_timestamp: bool = False
    seed: int = 0

    def __post_init__(self):
        if (
            isinstance(self.request_rate, bool)
            or math.isnan(self.request_rate)
            or self.request_rate <= 0
        ):
            raise ValueError("request_rate must be positive or infinity")
