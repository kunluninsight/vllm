from typing import Union

import numpy as np
from hisim.simulation.types import RequestStats, SchedulerConfig
from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo


def profile_device_available_bytes(
    model: ModelInfo, device: AcceleratorInfo, scheduler_config: SchedulerConfig
) -> int:
    from hisim.time_predictor.aiconfigurator import get_perf_model

    perf_model = get_perf_model(scheduler_config, model)
    weights = 0
    for op in perf_model.context_ops:
        weights += op.get_weights()
    # Count weights on a single GPU
    weights /= perf_model.config.pp_size
    framework_reserved_mem_gb = 1.4
    rest_memory = (
        scheduler_config.mem_fraction_static * device.hbm_capacity_gb
        - framework_reserved_mem_gb
    ) * (1 << 30) - weights
    return int(rest_memory)


def calc_metrics(requests: list[Union[RequestStats, dict]]) -> dict:
    requests = requests.copy()
    for idx, req in enumerate(requests):
        if isinstance(req, dict):
            requests[idx] = RequestStats(
                gen_token_latencies=req["gen_token_latencies"],
                queue_start=req["queue_start"],
                queue_end=req["queue_end"],
                last_event_time=req["last_event_time"],
                input_length=req["input_length"],
                output_length=req["output_length"],
                final_device_hit_len=req["final_device_hit_len"],
                final_host_hit_len=req["final_host_hit_len"],
                final_storage_hit_len=req["final_storage_hit_len"],
            )

    ttfts = []
    tpots = []
    itls = []
    e2e_latencies = []
    total_dur_s = 1e-9
    total_input = 0
    total_output = 0
    completed = 0
    total_reused_tokens = 0
    total_device_hit_tokens = 0
    total_host_hit_tokens = 0
    total_storage_hit_tokens = 0
    queue_durs = []
    for req in requests:
        if not req.is_complete():
            continue
        completed += 1
        if req.gen_token_latencies:
            # Requests finished without any recorded decode step (e.g.
            # connector-side completion) contribute no TTFT sample.
            ttfts.append(req.gen_token_latencies[0])
        queue_durs.append(req.queue_end - req.queue_start)
        if len(req.gen_token_latencies) > 1:
            # output length > 1
            tpots.append(np.mean(req.gen_token_latencies[1:]))
        itls.extend(req.gen_token_latencies[1:])
        e2e_latencies.append(sum(req.gen_token_latencies))
        total_dur_s = max(total_dur_s, req.last_event_time)
        total_input += req.input_length
        total_output += req.output_length
        total_reused_tokens += req.final_device_hit_len
        total_device_hit_tokens += req.final_device_hit_len - req.final_host_hit_len
        total_host_hit_tokens += req.final_host_hit_len - req.final_storage_hit_len
        total_storage_hit_tokens += req.final_storage_hit_len
    return {
        "num_requests": len(requests),
        "completed": completed,
        "total_input": total_input,
        "total_output": total_output,
        "duration": total_dur_s,
        "request_throughput": len(requests) / total_dur_s,
        "input_throughput": total_input / total_dur_s,
        "output_throughput": total_output / total_dur_s,
        "total_throughput": (total_input + total_output) / total_dur_s,
        "prefix_cache_reused_ratio": (
            0 if total_input == 0 else total_reused_tokens / total_input
        ),
        "kv_cache_storage_hit_ratio": (
            0 if total_input == 0 else total_storage_hit_tokens / total_input
        ),
        "kv_cache_host_hit_ratio": (
            0 if total_input == 0 else total_host_hit_tokens / total_input
        ),
        "kv_cache_device_hit_ratio": (
            0 if total_input == 0 else total_device_hit_tokens / total_input
        ),
        "mean_ttft_ms": np.mean(ttfts or 0) * 1000,
        "median_ttft_ms": np.median(ttfts or 0) * 1000,
        "std_ttft_ms": np.std(ttfts or 0) * 1000,
        "p90_ttft_ms": np.percentile(ttfts or 0, 90) * 1000,
        "p95_ttft_ms": np.percentile(ttfts or 0, 95) * 1000,
        "p99_ttft_ms": np.percentile(ttfts or 0, 99) * 1000,
        "mean_queue_ms": np.mean(queue_durs or 0) * 1000,
        "mean_tpot_ms": np.mean(tpots or 0) * 1000,
        "median_tpot_ms": np.median(tpots or 0) * 1000,
        "std_tpot_ms": np.std(tpots or 0) * 1000,
        "p90_tpot_ms": np.percentile(tpots or 0, 90) * 1000,
        "p95_tpot_ms": np.percentile(tpots or 0, 95) * 1000,
        "p99_tpot_ms": np.percentile(tpots or 0, 99) * 1000,
        "mean_itl_ms": np.mean(itls or 0) * 1000,
        "median_itl_ms": np.median(itls or 0) * 1000,
        "std_itl_ms": np.std(itls or 0) * 1000,
        "p90_itl_ms": np.percentile(itls or 0, 90) * 1000,
        "p95_itl_ms": np.percentile(itls or 0, 95) * 1000,
        "p99_itl_ms": np.percentile(itls or 0, 99) * 1000,
        "max_itl_ms": np.max(itls or 0) * 1000,
        "mean_e2e_latency_ms": np.mean(e2e_latencies) * 1000,
        "median_e2e_latency_ms": np.median(e2e_latencies) * 1000,
        "std_e2e_latency_ms": np.std(e2e_latencies) * 1000,
        "p90_e2e_latency_ms": np.percentile(e2e_latencies or 0, 90) * 1000,
        "p95_e2e_latency_ms": np.percentile(e2e_latencies or 0, 95) * 1000,
        "p99_e2e_latency_ms": np.percentile(e2e_latencies or 0, 99) * 1000,
        "time_cost": -1,  # Updated by external benchmark caller
    }
