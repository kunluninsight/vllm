# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulator collectives over the native EngineCore DP process group."""

import json

import torch


class DPContext:
    def __init__(self, rank: int, size: int):
        self.rank = rank
        self.size = size
        self.group = None

    def gather(self, values: torch.Tensor) -> list[torch.Tensor]:
        if self.size == 1:
            return [values]
        if self.group is None:
            raise RuntimeError("The native DP process group has not been bound")
        results = [torch.empty_like(values) for _ in range(self.size)]
        self.group.allgather([results], [values]).wait()
        return results

    def gather_json(self, value):
        if self.size == 1:
            return [value]
        data = json.dumps(value).encode()
        sizes = self.gather(torch.tensor([len(data)], dtype=torch.int64))
        buffer = torch.zeros(max(int(size[0]) for size in sizes), dtype=torch.uint8)
        buffer[: len(data)] = torch.tensor(list(data), dtype=torch.uint8)
        return [
            json.loads(bytes(item[: int(size[0])].tolist()))
            for item, size in zip(self.gather(buffer), sizes)
        ]

    def sum(self, values: torch.Tensor) -> None:
        if self.size > 1:
            if self.group is None:
                raise RuntimeError("The native DP process group has not been bound")
            self.group.allreduce([values]).wait()
