# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scheduling data and time-prediction contracts."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ScheduleRequest:
    extend_length: int = 0
    past_kv_length: int = 0
    prompt_length: int = 0
    # On a resumed request the prefill also includes earlier output tokens.
    prefill_length: int | None = None

    @property
    def num_prefill_tokens(self) -> int:
        return (
            self.prompt_length if self.prefill_length is None else self.prefill_length
        )


@dataclass
class ScheduleBatch:
    reqs: list[ScheduleRequest] = field(default_factory=list)
    rank: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    @property
    def num_tokens(self) -> int:
        return sum(req.extend_length for req in self.reqs)

    def is_empty(self) -> bool:
        return not self.reqs

    def is_decode(self) -> bool:
        return bool(self.reqs) and all(
            req.extend_length == 1 and req.past_kv_length >= req.num_prefill_tokens
            for req in self.reqs
        )


class TimePredictor(Protocol):
    def predict(self, batches: tuple[ScheduleBatch, ...]) -> float:
        """Return the complete synchronized step duration in seconds."""
        ...


class PerRankTimePredictor(ABC):
    @abstractmethod
    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        """Return forward latency in seconds for one rank."""

    def predict_sample_tokens_time(self, total_tokens: int) -> float:
        return 0.0

    def predict_step_time(self, batch: ScheduleBatch) -> tuple[float, float]:
        if batch.is_empty():
            return 0.0, 0.0
        return (
            self.predict_infer_time(batch),
            self.predict_sample_tokens_time(batch.num_tokens),
        )

    def predict(self, batches: tuple[ScheduleBatch, ...]) -> float:
        latencies = []
        for batch in batches:
            forward, sample = self.predict_step_time(batch)
            if not all(
                math.isfinite(value) and value >= 0 for value in (forward, sample)
            ):
                raise ValueError("Predicted latencies must be finite and non-negative")
            latencies.append(forward + sample)
        return max(latencies, default=0.0)
