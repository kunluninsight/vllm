# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSON-compatible KV layouts and TP redistribution plans."""

from dataclasses import dataclass

CONNECTOR_MODULE = "vllm_simulator.plugin.kv_transfer.disaggregated"
CONNECTOR_NAME = "SimulatorPDConnector"


def head_range(tp, heads, rank):
    if heads >= tp and heads % tp == 0:
        return rank * (heads // tp), (rank + 1) * (heads // tp)
    if tp > heads and tp % heads == 0:
        first = rank // (tp // heads)
        return first, first + 1
    raise ValueError("PD TP size must divide KV heads or replicate whole KV heads")


@dataclass(frozen=True)
class KVShardTransfer:
    source_rank: int
    destination_rank: int
    first_head: int
    num_heads: int
    num_bytes: int


@dataclass(frozen=True)
class WireKVLayout:
    instance: str
    tp_size: int
    num_kv_heads: int
    block_size: int
    block_bytes: int
    cache_layout: str | None
    physical_signature: str
    canonical_signature: str | None
    head_bytes: int
    model_signature: str = ""

    def plan_to(self, destination):
        if self.model_signature != destination.model_signature:
            raise ValueError("PD instances must use the same model and model dtype")
        if self.block_size != destination.block_size:
            raise ValueError("PD instances must use the same KV block size")
        if self.tp_size == destination.tp_size and (
            self.physical_signature == destination.physical_signature
            and self.cache_layout == destination.cache_layout
            and self.block_bytes == destination.block_bytes
        ):
            shards = []
            for rank in range(self.tp_size):
                first, end = head_range(self.tp_size, self.num_kv_heads, rank)
                shards.append(
                    KVShardTransfer(rank, rank, first, end - first, self.block_bytes)
                )
            return tuple(shards), False
        if (
            not self.canonical_signature
            or self.canonical_signature != destination.canonical_signature
            or self.num_kv_heads != destination.num_kv_heads
            or self.head_bytes != destination.head_bytes
        ):
            raise ValueError("PD instances have incompatible KV head layouts")
        shards = []
        for dst in range(destination.tp_size):
            begin, end = head_range(destination.tp_size, destination.num_kv_heads, dst)
            if self.tp_size > self.num_kv_heads:
                replicas = self.tp_size // self.num_kv_heads
                for head in range(begin, end):
                    src = head * replicas + dst % replicas
                    shards.append(KVShardTransfer(src, dst, head, 1, self.head_bytes))
            else:
                for src in range(self.tp_size):
                    src_begin, src_end = head_range(
                        self.tp_size, self.num_kv_heads, src
                    )
                    first, last = max(begin, src_begin), min(end, src_end)
                    if first < last:
                        shards.append(
                            KVShardTransfer(
                                src,
                                dst,
                                first,
                                last - first,
                                (last - first) * self.head_bytes,
                            )
                        )
        return tuple(shards), True
