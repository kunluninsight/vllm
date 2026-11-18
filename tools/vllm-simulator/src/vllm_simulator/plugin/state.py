# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request metadata and clocks owned by one simulator engine."""

import json
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayMetadata:
    created_time: float
    dispatch_at: float
    total_requests: int | None
    conversation_id: str | None = None
    turn: int | None = None
    abort_round_on_cancel: bool = False

    @classmethod
    def from_extra_args(cls, extra_args, now: float):
        metadata = (extra_args or {}).get("simulation", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            raise ValueError("simulation must be a JSON object")
        created = float(metadata.get("created_time", now))
        dispatch = float(metadata.get("dispatch_at", created))
        total = metadata.get("total_requests")
        if (
            not math.isfinite(created)
            or created < 0
            or not math.isfinite(dispatch)
            or dispatch < created
            or (total is not None and (type(total) is not int or total <= 0))
        ):
            raise ValueError("Invalid simulation arrival times or total_requests")
        conversation = metadata.get("conversation_id")
        turn = metadata.get("turn")
        if (conversation is not None or turn is not None) and (
            not isinstance(conversation, str)
            or not conversation
            or type(turn) is not int
            or turn < 0
        ):
            raise ValueError(
                "conversation_id and a non-negative turn must be supplied together"
            )
        abort_round = metadata.get("abort_round_on_cancel", False)
        if not isinstance(abort_round, bool):
            raise ValueError("abort_round_on_cancel must be a boolean")
        return cls(created, dispatch, total, conversation, turn, abort_round)


@dataclass
class VirtualClock:
    now: float = 0.0
    device_available_at: float = 0.0

    def reset(self) -> None:
        self.now = self.device_available_at = 0.0

    def wait_until(self, deadline: float, timeout: float | None = None) -> None:
        self.now = max(self.now, deadline)


@dataclass
class MonotonicClock:
    device_available_at: float = 0.0

    @property
    def now(self) -> float:
        return time.monotonic()

    def wait_until(self, deadline: float, timeout: float | None = None) -> None:
        remaining = deadline - self.now
        if remaining <= 0:
            return
        if timeout is not None and timeout < remaining:
            if timeout > 0:
                time.sleep(timeout)
            if self.now < deadline:
                raise TimeoutError("Simulated batch is still executing")
        else:
            time.sleep(remaining)
