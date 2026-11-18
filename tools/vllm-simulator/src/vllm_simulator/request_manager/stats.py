# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record request timings without controlling admission."""


class RequestStats:
    def __init__(self, mode):
        self.mode = mode
        self.records = {}

    def add(self, item):
        meta = item.metadata
        self.records[item.request.request_id] = {
            "mode": self.mode,
            "created_time": meta.created_time,
            "admitted_at": None,
            "first_scheduled_at": None,
            "last_event_time": meta.created_time,
            "ttft": None,
            "itls": [],
            "input_length": item.request.num_prompt_tokens,
            "local_kv_hit_len": 0,
            "ext_kv_hit_len": 0,
        }

    def admit(self, rid, now):
        self.records[rid]["admitted_at"] = now

    def discard(self, rid):
        self.records.pop(rid, None)

    def reset(self):
        self.records.clear()

    def update(self, step, outputs):
        now = step.finished_at
        for rid in step.scheduled_ids:
            stats = self.records.get(rid)
            if stats is not None and stats["first_scheduled_at"] is None:
                stats["first_scheduled_at"] = step.started_at
        finished = []
        for batch in outputs.values():
            for output in batch.outputs:
                stats = self.records.get(output.request_id)
                if stats is None:
                    continue
                if output.prefill_stats is not None:
                    stats["local_kv_hit_len"] = (
                        output.prefill_stats.num_local_cached_tokens
                    )
                    stats["ext_kv_hit_len"] = (
                        output.prefill_stats.num_external_cached_tokens
                    )
                for _ in output.new_token_ids:
                    latency = now - stats["last_event_time"]
                    if stats["ttft"] is None:
                        stats["ttft"] = latency
                    else:
                        stats["itls"].append(latency)
                    stats["last_event_time"] = now
                if output.finished:
                    stats["output_length"] = int(stats["ttft"] is not None) + len(
                        stats["itls"]
                    )
                    if step.dp_rank is not None:
                        stats["dp_rank"] = step.dp_rank
                        stats["step_batch_tokens"] = list(step.batch_tokens)
                    output.kv_transfer_params = {
                        **(output.kv_transfer_params or {}),
                        "simulation": stats,
                    }
                    self.records.pop(output.request_id)
                    finished.append(output.request_id)
        return finished
