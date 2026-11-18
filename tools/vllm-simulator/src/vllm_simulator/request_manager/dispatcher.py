# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission policies independent of EngineCore and its transport."""

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_simulator.plugin.state import ReplayMetadata

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass
class PendingRequest:
    request: "Request"
    metadata: ReplayMetadata


@dataclass(frozen=True, order=True)
class DispatchCandidate:
    dispatch_at: float
    order: int
    request_id: str


class RequestDispatcher(ABC):
    sync_metadata = False

    def __init__(self, max_concurrency: int | None = None):
        if max_concurrency is not None and (
            type(max_concurrency) is not int or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")
        self.max_concurrency = max_concurrency

    def validate_request(self, metadata: ReplayMetadata) -> None:
        return None

    def round_metadata(self) -> list:
        return []

    def validate_round(self, records: list) -> None:
        return None

    def drain_events(self) -> list:
        return []

    def apply_events(self, events: list) -> None:
        return None

    @property
    @abstractmethod
    def pending_count(self) -> int:
        """Number of requests waiting for admission."""

    @abstractmethod
    def add(self, item: PendingRequest) -> None:
        """Track a validated request."""

    def next_time(self) -> float:
        return float("inf")

    def candidates(self, now: float) -> list[DispatchCandidate]:
        return []

    def take(self, rid: str) -> PendingRequest:
        raise KeyError(rid)

    def cancel_pending(self) -> list["Request"]:
        return []

    @abstractmethod
    def complete(self, rid: str) -> None:
        """Record completion or cancellation."""

    @abstractmethod
    def reset(self) -> None:
        """Discard round state."""


class BlockingDispatcher(RequestDispatcher):
    """Track live requests admitted immediately by the native Scheduler."""

    def __init__(self):
        super().__init__()
        self._active: set[str] = set()

    @property
    def pending_count(self) -> int:
        return 0

    def add(self, item: PendingRequest) -> None:
        rid = item.request.request_id
        if rid in self._active:
            raise ValueError("Request ID is already active")
        self._active.add(rid)

    def complete(self, rid: str) -> None:
        self._active.discard(rid)

    def reset(self) -> None:
        self._active.clear()


class QueuedRequestDispatcher(RequestDispatcher):
    """Default timestamp queue for replay policies."""

    def __init__(self, max_concurrency: int | None = None):
        super().__init__(max_concurrency)
        self._pending: dict[str, PendingRequest] = {}
        self._records: dict[str, ReplayMetadata] = {}
        self._queue: list[DispatchCandidate] = []
        self._sequence = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def add(self, item: PendingRequest) -> None:
        self.validate_request(item.metadata)
        rid = item.request.request_id
        if rid in self._records:
            raise ValueError("Request IDs must be unique within a replay round")
        self._records[rid] = item.metadata
        self._pending[rid] = item
        heapq.heappush(
            self._queue,
            DispatchCandidate(item.metadata.dispatch_at, self._sequence, rid),
        )
        self._sequence += 1

    @abstractmethod
    def eligible(self, item: PendingRequest) -> bool:
        """Whether the request's dependencies have completed."""

    def next_time(self) -> float:
        return min(
            (
                item.metadata.dispatch_at
                for item in self._pending.values()
                if self.eligible(item)
            ),
            default=float("inf"),
        )

    def candidates(self, now: float) -> list[DispatchCandidate]:
        ready, parked = [], []
        while self._queue and self._queue[0].dispatch_at <= now:
            entry = heapq.heappop(self._queue)
            item = self._pending.get(entry.request_id)
            if item is None:
                continue
            parked.append(entry)
            if self.eligible(item):
                ready.append(entry)
                if self.max_concurrency and len(ready) >= self.max_concurrency:
                    break
        for entry in parked:
            heapq.heappush(self._queue, entry)
        return ready

    def take(self, rid: str) -> PendingRequest:
        return self._pending.pop(rid)

    def complete(self, rid: str) -> None:
        self._pending.pop(rid, None)

    def cancel_pending(self) -> list["Request"]:
        requests = [item.request for item in self._pending.values()]
        self._pending.clear()
        return requests

    def reset(self) -> None:
        self._pending.clear()
        self._records.clear()
        self._queue.clear()
        self._sequence = 0


class TimeDispatcher(QueuedRequestDispatcher):
    def eligible(self, item: PendingRequest) -> bool:
        return True

    def next_time(self) -> float:
        while self._queue and self._queue[0].request_id not in self._pending:
            heapq.heappop(self._queue)
        return self._queue[0].dispatch_at if self._queue else float("inf")


class ConcurrencyDispatcher(TimeDispatcher):
    def __init__(self, max_concurrency: int):
        super().__init__(max_concurrency)


class ConversationDispatcher(QueuedRequestDispatcher):
    sync_metadata = True

    def __init__(self, max_concurrency: int | None = None):
        super().__init__(max_concurrency)
        self.completed: set[tuple[str, int]] = set()
        self.events: list[tuple[str, int]] = []

    def validate_request(self, metadata: ReplayMetadata) -> None:
        if (
            metadata.conversation_id is None
            or metadata.turn is None
            or metadata.total_requests is None
        ):
            raise ValueError(
                "Conversation dispatch requires conversation_id, turn "
                "and total_requests"
            )

    def eligible(self, item: PendingRequest) -> bool:
        meta = item.metadata
        return meta.turn == 0 or (meta.conversation_id, meta.turn - 1) in self.completed

    def complete(self, rid: str) -> None:
        super().complete(rid)
        metadata = self._records.get(rid)
        if metadata is not None:
            self.events.append((metadata.conversation_id, metadata.turn))

    def round_metadata(self) -> list:
        return [
            [metadata.conversation_id, metadata.turn]
            for metadata in self._records.values()
        ]

    def drain_events(self) -> list:
        events, self.events = self.events, []
        return events

    def apply_events(self, events: list) -> None:
        self.completed.update((conversation, turn) for conversation, turn in events)

    def validate_round(self, records: list[list]) -> None:
        turns = {}
        for conversation, turn in records:
            seen = turns.setdefault(conversation, set())
            if turn in seen:
                raise ValueError("Duplicate conversation turn")
            seen.add(turn)
        if any(sorted(seen) != list(range(len(seen))) for seen in turns.values()):
            raise ValueError("Conversation turns must be contiguous and start at zero")

    def reset(self) -> None:
        super().reset()
        self.completed.clear()
        self.events.clear()
