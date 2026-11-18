# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in debug timeline using existing frontend outputs and stats."""

import gzip
import json
from functools import wraps
from pathlib import Path

from vllm.v1.engine import EngineCoreEventType
from vllm.v1.metrics.loggers import AggregateStatLoggerBase


class RequestTimeline(AggregateStatLoggerBase):
    def __init__(self, vllm_config, engine_indexes):
        config = vllm_config.additional_config["simulator"]["collector"]
        compressed = config.get("gzip", True)
        self.path = Path(config.get("output_dir", ".")) / (
            "requests.jsonl.gz" if compressed else "requests.jsonl"
        )
        self.rows = None
        self.pending = {}
        self.finished = set()
        self.processor = None

    def attach(self, processor):
        self.processor = processor
        original = processor.process_outputs

        @wraps(original)
        def process_outputs(outputs, engine_core_timestamp=None, *args, **kwargs):
            if self.rows is not None:
                for output in outputs:
                    state = processor.request_states.get(output.request_id)
                    if state is None:
                        continue
                    if state.parent_req is not None:
                        raise ValueError("request_timeline requires n=1")
                    self.observe(output, engine_core_timestamp)
            return original(outputs, engine_core_timestamp, *args, **kwargs)

        processor.process_outputs = process_outputs

    def observe(self, output, timestamp):
        rid = output.request_id
        if rid in self.finished:
            raise ValueError("request_timeline requires unique request IDs per window")
        row = self.pending.setdefault(
            rid,
            dict(
                queued_at=None,
                first_scheduled_at=None,
                first_token_at=None,
                last_token_at=None,
            ),
        )
        for event in output.events or ():
            if event.type == EngineCoreEventType.QUEUED and row["queued_at"] is None:
                row["queued_at"] = event.timestamp
            elif (
                event.type == EngineCoreEventType.SCHEDULED
                and row["first_scheduled_at"] is None
            ):
                row["first_scheduled_at"] = event.timestamp
        if output.new_token_ids:
            if timestamp is None:
                raise ValueError("Token output lacks an EngineCore timestamp")
            if row["first_token_at"] is None:
                row["first_token_at"] = timestamp
            row["last_token_at"] = timestamp

    def start(self):
        if self.rows is not None:
            raise RuntimeError("Request collection is already active")
        if self.processor.get_num_unfinished_requests():
            raise RuntimeError("Start request_timeline only on an idle engine")
        self.pending.clear()
        self.finished.clear()
        self.rows = []

    def record(
        self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0
    ):
        if self.rows is not None and iteration_stats is not None:
            for r in iteration_stats.finished_requests:
                if r.request_id in self.finished:
                    raise ValueError("request_timeline requires unique request IDs")
                self.finished.add(r.request_id)
                times = self.pending.pop(r.request_id, {})
                self.rows.append(
                    {
                        "request_id": r.request_id,
                        "external_request_id": r.request_id,
                        "engine_index": engine_idx,
                        "clock": "engine_core_monotonic_seconds",
                        "queued_at": times.get("queued_at"),
                        "first_scheduled_at": times.get("first_scheduled_at"),
                        "first_token_at": times.get("first_token_at"),
                        "last_token_at": times.get("last_token_at"),
                        "prompt_tokens": r.num_prompt_tokens,
                        "output_tokens": r.num_generation_tokens,
                        "finish_reason": str(r.finish_reason),
                    }
                )

    def stop(self):
        if self.rows is None:
            raise RuntimeError("Request collection is not active")
        if self.pending or self.processor.get_num_unfinished_requests():
            raise RuntimeError(
                "Finish benchmark traffic before stopping request_timeline"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "wt") as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + "\n")
        self.rows = None

    def log_engine_initialized(self):
        pass
