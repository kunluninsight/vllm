# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linear per-rank step latency, reduced over the DP critical path."""

import math

from .base import PerRankTimePredictor, ScheduleBatch


class LinearTimePredictor(PerRankTimePredictor):
    def __init__(self, base_ms: float = 10.0, per_token_ms: float = 0.0):
        for value in (base_ms, per_token_ms):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Linear coefficients must be finite and non-negative")
        self.base_s = base_ms / 1000
        self.per_token_s = per_token_ms / 1000

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0
        return self.base_s + self.per_token_s * sum(r.extend_length for r in batch.reqs)
