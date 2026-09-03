"""Online serving benchmark client for the vLLM simulator.

Hijacks ``vllm.benchmarks.serve`` (the implementation behind
``vllm bench serve``) so that:

1. Traffic is replayed logically instead of in real time: the client never
   sleeps between requests. Every request carries ``simulation.created_time``
   (its logical arrival offset in seconds) and the server replays the traffic
   pattern from that timestamp (see ``C_VLLMSchedulerHook``).
2. Reported metrics come from the simulator backend: after the run, the
   per-request stats dumped by the server-side profile hook
   (``request.jsonl``) are aggregated with ``calc_metrics`` and substituted
   into the benchmark result.

Usage:
    python -m vllm_simulator.simulation.bench_serving [vllm bench serve args...]

The server must be the simulator server (see
``vllm_simulator.simulation.vllm.launch_server``). The profile hook uses
/start_profile and /stop_profile as simulation round markers (dump + reset);
the EngineArgs hook internally defaults the profiler to "cuda" so those
routes always exist, and the hijacked ``EngineCore.profile`` never runs a
real profiler.
"""

import argparse
import json
import os
import re
from typing import AsyncGenerator, Literal, Optional

import aiohttp
import numpy as np
from vllm.benchmarks import serve
from vllm.benchmarks.datasets import SampleRequest

from vllm_simulator.simulation.manager import Envs
from vllm_simulator.simulation.utils import calc_metrics

# Hijack aiohttp request sending to rewrite simulation metadata into a
# server-visible path.
#
# ``override_get_request`` attaches ``simulation`` to each request through
# ``SampleRequest.request_overrides``, so it lands in the request payload as a
# top-level field. vLLM's OpenAI protocol keeps unknown top-level fields as
# ignored extras; only ``vllm_xargs`` is forwarded into
# ``SamplingParams.extra_args``, which is where the server-side scheduler hook
# reads the metadata. ``vllm_xargs`` values are restricted to scalars/lists by
# pydantic, so the nested dict is JSON-encoded; the server hook decodes it.
_ORIG_AIOHTTP_REQUEST = None


def install_aiohttp_json_hijack(
    *,
    hijack_url_regex: Optional[str],
) -> None:
    global _ORIG_AIOHTTP_REQUEST
    if _ORIG_AIOHTTP_REQUEST is not None:
        return

    pattern = re.compile(hijack_url_regex) if hijack_url_regex else None
    _ORIG_AIOHTTP_REQUEST = aiohttp.ClientSession._request

    async def _patched_request(self, method, url, **kwargs):
        if pattern is not None and pattern.search(url):
            payload = kwargs.get("json", None)
            if isinstance(payload, dict) and "simulation" in payload:
                vllm_xargs = payload.setdefault("vllm_xargs", {})
                vllm_xargs["simulation"] = json.dumps(payload.pop("simulation"))
                kwargs["json"] = payload

        return await _ORIG_AIOHTTP_REQUEST(self, method, url, **kwargs)

    aiohttp.ClientSession._request = _patched_request


# Override request generation for simulation mode.
#
# Same delay model as the native ``serve.get_request`` (gamma arrivals with
# burstiness, optional ramp-up, self-timed traces), but the client never
# sleeps: the cumulative delay becomes ``simulation.created_time`` on each
# request, and the simulator replays the traffic pattern against its (virtual)
# clock. Self-timed traces are normalized relative to the first request so the
# simulation clock starts at zero.
async def override_get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
    self_timed: bool = False,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    if not isinstance(input_requests, list):
        input_requests = list(input_requests)

    total_requests = len(input_requests)
    assert total_requests > 0, "No requests provided."

    request_rates: list[float] = []
    delay_ts: list[float] = []

    if self_timed:
        # Sort by timestamp for correct replay, then normalize relative to
        # the first request.
        input_requests.sort(key=lambda r: r.timestamp or 0.0)
        first_ts = input_requests[0].timestamp or 0.0
        for request in input_requests:
            delay_ts.append((request.timestamp or 0.0) - first_ts)
            # No notion of RPS for self-timed traces, same as native.
            request_rates.append(0.0)
    else:
        for request_index in range(total_requests):
            # Reuse the native helper so ramp-up semantics stay in sync.
            current_request_rate = serve._get_current_request_rate(
                ramp_up_strategy,
                ramp_up_start_rps,
                ramp_up_end_rps,
                request_index,
                total_requests,
                request_rate,
            )
            assert current_request_rate > 0.0, (
                f"Obtained non-positive request rate {current_request_rate}."
            )
            request_rates.append(current_request_rate)
            if current_request_rate == float("inf"):
                delay_ts.append(0)
            elif burstiness == float("inf"):
                # Constant inter-arrival time in the infinite-burstiness limit.
                delay_ts.append(1.0 / current_request_rate)
            else:
                theta = 1.0 / (current_request_rate * burstiness)
                delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

        # Cumulative arrival offsets from the first request.
        for i in range(1, len(delay_ts)):
            delay_ts[i] += delay_ts[i - 1]
        if ramp_up_strategy is None and delay_ts[-1] != 0:
            # Same stabilization as the native generator: close the gap
            # between the sampled total delay and the target duration so
            # throughput is stable across random seeds.
            target_total_delay_s = total_requests / request_rate
            normalize_factor = target_total_delay_s / delay_ts[-1]
            delay_ts = [delay * normalize_factor for delay in delay_ts]

    for request_index, request in enumerate(input_requests):
        overrides = dict(request.request_overrides or {})
        overrides["simulation"] = {
            "created_time": delay_ts[request_index],
            "total_request": total_requests,
        }
        request.request_overrides = overrides
        yield request, request_rates[request_index]


# Replace benchmark-side metrics with backend-generated simulation metrics.
#
# Client-side metrics describe the local execution path of the benchmark tool,
# not the simulated execution on the backend. The server-side profile hook
# dumps per-request stats to ``request.jsonl`` on /stop_profile, which
# ``serve.benchmark`` issues right before returning, so the wrapped benchmark
# can aggregate them and override the result.
#
# If the backend stats file is missing, the client-side result is kept.
_BASE_RESULT_KEYS = {
    "duration": "duration",
    "completed": "completed",
    "total_input": "total_input_tokens",
    "total_output": "total_output_tokens",
    "request_throughput": "request_throughput",
    "output_throughput": "output_throughput",
    "total_throughput": "total_token_throughput",
}

# vLLM serve result name -> calc_metrics name.
_LATENCY_METRICS = {
    "ttft": "ttft",
    "tpot": "tpot",
    "itl": "itl",
    "e2el": "e2e_latency",
}

# Simulation-only stats worth keeping in the result json.
_EXTRA_RESULT_KEYS = (
    "num_requests",
    "input_throughput",
    "mean_queue_ms",
    "max_itl_ms",
    "prefix_cache_reused_ratio",
    "kv_cache_device_hit_ratio",
    "kv_cache_host_hit_ratio",
    "kv_cache_storage_hit_ratio",
)


def _load_simulation_metrics() -> dict | None:
    """Aggregate per-request stats dumped by the server-side profile hook."""
    stats_path = os.path.join(Envs.output_dir(), "request.jsonl")
    if not os.path.exists(stats_path):
        return None
    with open(stats_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return None
    return calc_metrics(rows)


def _apply_simulation_metrics(result: dict, sim: dict) -> None:
    for sim_key, result_key in _BASE_RESULT_KEYS.items():
        result[result_key] = sim[sim_key]
    for vllm_name, sim_name in _LATENCY_METRICS.items():
        # Percentile keys exist only when selected via --percentile-metrics /
        # --metric-percentiles; override the ones present in the result.
        for stat in ("mean", "median", "std", "p90", "p95", "p99"):
            result_key = f"{stat}_{vllm_name}_ms"
            sim_key = f"{stat}_{sim_name}_ms"
            if result_key in result and sim_key in sim:
                result[result_key] = sim[sim_key]
        # calc_metrics reports the median instead of p50.
        p50_key = f"p50_{vllm_name}_ms"
        if p50_key in result:
            result[p50_key] = sim[f"median_{sim_name}_ms"]
    for key in _EXTRA_RESULT_KEYS:
        if key in sim:
            result[key] = sim[key]


original_benchmark = serve.benchmark


async def wrapped_benchmark(*args, **kwargs):
    result = await original_benchmark(*args, **kwargs)

    sim_metrics = _load_simulation_metrics()
    if sim_metrics is None:
        print(
            "[simulation] No backend request stats found under "
            f"{Envs.output_dir()}, keeping client-side metrics. Launch the "
            "server with a profiler configured so that /start_profile and "
            "/stop_profile are registered."
        )
        return result

    _apply_simulation_metrics(result, sim_metrics)

    print("{s:{c}^{n}}".format(s=" Simulation Metrics ", n=50, c="="))
    print("{:<40} {:<10}".format("Completed requests:", sim_metrics["completed"]))
    print("{:<40} {:<10.2f}".format("Simulated duration (s):", sim_metrics["duration"]))
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", sim_metrics["output_throughput"]
        )
    )
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", sim_metrics["mean_ttft_ms"]))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", sim_metrics["mean_tpot_ms"]))
    print(
        "{:<40} {:<10.2f}".format("Mean E2EL (ms):", sim_metrics["mean_e2e_latency_ms"])
    )
    print("=" * 50)
    return result


original_main_async = serve.main_async


async def wrapped_main_async(args: argparse.Namespace):
    # /start_profile and /stop_profile double as simulation round markers on
    # the server (stats dump + reset), so profiling is always on.
    args.profile = True
    return await original_main_async(args)


serve.get_request = override_get_request
serve.benchmark = wrapped_benchmark
serve.main_async = wrapped_main_async

install_aiohttp_json_hijack(hijack_url_regex=r"completions$")


if __name__ == "__main__":
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM simulator online serving benchmark "
        "(hijacked `vllm bench serve`)"
    )
    serve.add_cli_args(parser)
    args = parser.parse_args()
    serve.main(args)
