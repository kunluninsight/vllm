# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect CUDA event durations without synchronizing the serving hot path."""

import gzip
import json
import time
from collections import deque
from pathlib import Path

import torch

from vllm.v1.worker.gpu_worker import Worker


class EventTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def seconds(self):
        return self.start.elapsed_time(self.end) / 1000


def batch_requests(output):
    """Snapshot scheduled lengths before the model runner updates its state."""
    lengths = {
        rid: (past, generated)
        for rid, past, generated in zip(
            output.scheduled_cached_reqs.req_ids,
            output.scheduled_cached_reqs.num_computed_tokens,
            output.scheduled_cached_reqs.num_output_tokens,
        )
    }
    lengths.update(
        (
            req.req_id,
            (req.num_computed_tokens, len(req.prefill_token_ids) - req.prompt_len),
        )
        for req in output.scheduled_new_reqs
    )
    return [
        (rid, extend, *lengths[rid])
        for rid, extend in output.num_scheduled_tokens.items()
    ]


class CollectorWorker(Worker):
    """Native GPU execution with explicit, repeatable collection windows.

    Select using `--worker-cls` and enable the `simulator_collector` endpoint
    plugin. Tensor parallelism is supported; PP, DP and speculative decoding
    are outside this collector's scope.
    """

    def init_device(self):
        self._collector_config = self.vllm_config.additional_config.get(
            "simulator", {}
        ).get("collector", {})
        self._collection = None
        self._pending_forward = None
        self._timers = deque()
        if (
            self.parallel_config.pipeline_parallel_size != 1
            or self.parallel_config.data_parallel_size != 1
            or self.parallel_config.prefill_context_parallel_size != 1
            or self.parallel_config.decode_context_parallel_size != 1
            or self.vllm_config.speculative_config is not None
        ):
            raise ValueError(
                "CollectorWorker requires PP=DP=PCP=DCP=1 and no speculative decode"
            )
        super().init_device()

    def start_collection(self):
        if self._collection is not None:
            raise RuntimeError("Collection is already active")
        directory = Path(self._collector_config.get("output_dir", "."))
        compressed = self._collector_config.get("gzip", True)
        self._collection_path = directory / (
            f"rank{self.rank}.schedule_batch.jsonl" + (".gz" if compressed else "")
        )
        self._collection = []
        self._step_index = 0
        return str(self._collection_path)

    def _drain(self, *, wait=False):
        while self._timers:
            row, forward, sample = self._timers[0]
            last = sample if sample is not None else forward
            if wait:
                last.end.synchronize()
            elif not last.end.query():
                break
            row["iter_latency"] = forward.seconds()
            row["sample_tokens_latency"] = sample.seconds() if sample else 0.0
            row["end_timestamp"] = row["start_timestamp"] + row["iter_latency"]
            self._collection.append(row)
            self._timers.popleft()

    def stop_collection(self):
        if self._collection is None:
            raise RuntimeError("Collection is not active")
        if self._pending_forward is not None:
            raise RuntimeError("Stop collection after all requests have completed")
        self._drain(wait=True)
        self._collection_path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if self._collector_config.get("gzip", True) else open
        with opener(self._collection_path, "wt") as stream:
            for row in self._collection:
                stream.write(json.dumps(row) + "\n")
        steps = len(self._collection)
        self._collection = None
        return {"path": str(self._collection_path), "steps": steps}

    def execute_model(self, scheduler_output):
        if self._collection is None or not scheduler_output.num_scheduled_tokens:
            return super().execute_model(scheduler_output)
        self._drain()
        requests = batch_requests(scheduler_output)
        row = {
            "step_id": self._step_index,
            "start_timestamp": time.time(),
            "worker_start_monotonic_s": time.monotonic(),
            "requests": requests,
            "forward_mode": 2 if all(req[1] == 1 for req in requests) else 1,
            "total_tokens": sum(req[1] for req in requests),
        }
        self._step_index += 1
        timer = EventTimer()
        timer.start.record()
        result = super().execute_model(scheduler_output)
        timer.end.record()
        if result is None:
            self._pending_forward = (row, timer)
        else:
            self._timers.append((row, timer, None))
        return result

    def sample_tokens(self, grammar_output):
        pending = self._pending_forward
        if pending is None:
            return super().sample_tokens(grammar_output)
        self._pending_forward = None
        timer = EventTimer()
        timer.start.record()
        result = super().sample_tokens(grammar_output)
        timer.end.record()
        self._timers.append((*pending, timer))
        return result

    def shutdown(self):
        self._collection = None
        self._pending_forward = None
        self._timers.clear()
        super().shutdown()
