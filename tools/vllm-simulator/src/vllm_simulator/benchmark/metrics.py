# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics derived exclusively from virtual-time request records."""

import numpy as np


def calc_metrics(
    requests: list[dict], duration_start_time: float | None = None
) -> dict:
    start = (
        min((r["created_time"] for r in requests), default=0.0)
        if duration_start_time is None
        else duration_start_time
    )
    duration = max((r["last_event_time"] for r in requests), default=start) - start
    total_input = sum(r["input_length"] for r in requests)
    total_output = sum(r["output_length"] for r in requests)
    local = sum(r.get("local_kv_hit_len", 0) for r in requests)
    external = sum(r.get("ext_kv_hit_len", 0) for r in requests)
    metrics = {
        "num_requests": len(requests),
        "completed": len(requests),
        "total_input": total_input,
        "total_output": total_output,
        "duration": duration,
        "request_throughput": len(requests) / duration if duration else 0.0,
        "input_throughput": total_input / duration if duration else 0.0,
        "output_throughput": total_output / duration if duration else 0.0,
        "total_throughput": (total_input + total_output) / duration
        if duration
        else 0.0,
        "prompt_tokens_cached_ratio": (local + external) / total_input
        if total_input
        else 0.0,
        "prompt_tokens_local_cache_hit_ratio": local / total_input
        if total_input
        else 0.0,
        "prompt_tokens_external_kv_transfer_ratio": external / total_input
        if total_input
        else 0.0,
        "mean_queue_ms": float(
            np.mean([r["first_scheduled_at"] - r["admitted_at"] for r in requests])
        )
        * 1000
        if requests
        else 0.0,
    }
    itls = [r["itls"] for r in requests]
    groups = {
        "ttft": [r["ttft"] for r in requests if r["ttft"] is not None],
        "tpot": [sum(times) / len(times) for times in itls if times],
        "itl": [value for times in itls for value in times],
        "e2e_latency": [r["last_event_time"] - r["created_time"] for r in requests],
    }
    for name, values in groups.items():
        series = np.array(values or [0.0]) * 1000
        metrics.update(
            {
                f"mean_{name}_ms": float(np.mean(series)),
                f"median_{name}_ms": float(np.median(series)),
                f"std_{name}_ms": float(np.std(series)),
            }
        )
        for percentile in (90, 95, 99):
            metrics[f"p{percentile}_{name}_ms"] = float(
                np.percentile(series, percentile)
            )
    metrics["max_itl_ms"] = max(groups["itl"], default=0.0) * 1000
    return metrics
