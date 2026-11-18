# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-time block transfers between explicitly identified metadata regions."""

from dataclasses import dataclass
from typing import Literal

import torch

Direction = Literal["load", "store"]
TransferId = tuple[Direction, int]


@dataclass(frozen=True)
class BlockSelection:
    region: str
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class TransferJob:
    direction: Direction
    job_id: int
    source: BlockSelection
    destination: BlockSelection
    num_bytes: int

    @property
    def key(self) -> TransferId:
        return self.direction, self.job_id


@dataclass(frozen=True)
class TransferResult:
    job: TransferJob
    duration: float = 0.0


class TransferBackend:
    """Keep layouts and completion state; native managers own cache residency."""

    def __init__(self):
        self.regions: dict[str, torch.Tensor] = {}
        self._pending: dict[TransferId, TransferJob] = {}
        self._completed: dict[TransferId, TransferResult] = {}

    def register_region(self, name: str, num_blocks: int, block_bytes: int) -> None:
        if name in self.regions:
            raise ValueError(f"Transfer region {name!r} is already registered")
        if num_blocks < 0 or block_bytes <= 0:
            raise ValueError(
                "Transfer region sizes must be non-negative with positive block bytes"
            )
        self.regions[name] = torch.empty(
            (num_blocks, block_bytes), dtype=torch.uint8, device="meta"
        )

    def submit(self, job: TransferJob) -> None:
        if job.key in self._pending or job.key in self._completed:
            raise ValueError(f"Transfer {job.key} is already submitted")
        if job.num_bytes < 0:
            raise ValueError("Transfer byte count must be non-negative")
        for selection in (job.source, job.destination):
            layout = self.regions[selection.region]
            if any(
                block < 0 or block >= layout.shape[0] for block in selection.block_ids
            ):
                raise ValueError(
                    f"Transfer block is outside region {selection.region!r}"
                )
            if job.num_bytes > len(selection.block_ids) * layout.shape[1]:
                raise ValueError("Transfer exceeds the selected block capacity")
        self._pending[job.key] = job

    def wait(self, job_ids: set[TransferId]) -> None:
        for key in job_ids:
            job = self._pending.pop(key, None)
            if job is not None:
                self._completed[key] = TransferResult(job)

    def wait_all(self) -> None:
        self.wait(set(self._pending))

    def poll(self) -> tuple[TransferResult, ...]:
        self.wait_all()
        results = tuple(self._completed.values())
        self._completed.clear()
        return results

    def shutdown(self) -> None:
        self._pending.clear()
        self._completed.clear()
        self.regions.clear()
