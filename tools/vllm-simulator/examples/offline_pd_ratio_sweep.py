# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Approximate P:D allocation by composing independent virtual-time replays.

With fixed arrivals, round-robin routing and no D-to-P feedback, P completion
events can be computed before replaying D. Each D arrival is its P completion
time plus a fixed transfer delay. Keeping the same virtual time origin preserves
P/D pipeline overlap even though the phases execute sequentially in wall time.
Offline snapshots retain native D KV allocation but omit transfer contention,
P KV retention and backpressure. A full snapshot still replays the last prompt
token for sampling, and P's output token is discarded.

See `docs/offline-pd-ratio-analysis.md` in the simulator directory for the
derivation, workload format, throughput accounting and approximation limits.
"""

import argparse
import asyncio
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ["VLLM_PLUGINS"] = "simulator"

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import tokens_input
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

OFFLINE_CONNECTOR = {
    "kv_connector": "SimulatorOfflineKVConnector",
    "kv_connector_module_path": "vllm_simulator.plugin.kv_transfer.offline",
    "kv_role": "kv_consumer",
}


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    prompt_token_ids: list[int]
    output_length: int
    created_time: float

    def __post_init__(self):
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or not self.prompt_token_ids
            or any(
                type(token) is not int or token < 0 for token in self.prompt_token_ids
            )
            or type(self.output_length) is not int
            or self.output_length <= 0
            or not math.isfinite(self.created_time)
            or self.created_time < 0
        ):
            raise ValueError(
                "Invalid request ID, token IDs, output length or arrival time"
            )


async def replay_partition(engine, partition, *, decode=False):
    """Replay one cold logical instance, retaining the shared trace time origin."""
    if not partition:
        return []
    if not await engine.reset_prefix_cache(reset_connector=True):
        raise RuntimeError("Cannot reset the logical instance's KV pool")

    async def generate(request):
        extra = {
            "simulation": {
                "created_time": request.created_time,
                "total_requests": len(partition),
            }
        }
        if decode:
            extra["kv_transfer_params"] = {
                "offline_kv_tokens": len(request.prompt_token_ids)
            }
        params = SamplingParams(
            max_tokens=request.output_length if decode else 1,
            temperature=0,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            extra_args=extra,
        )
        async for result in engine.generate(
            tokens_input(request.prompt_token_ids), params, request.request_id
        ):
            if result.finished:
                return result.kv_transfer_params["simulation"]
        raise RuntimeError(f"No final result for {request.request_id}")

    return await asyncio.gather(*(generate(request) for request in partition))


async def sweep(requests, ratios, p_args, d_args, transfer_delay_s=0):
    """Run all P traces, then D traces; return metrics and request records."""
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("The dataset must be nonempty and request IDs must be unique")
    if not math.isfinite(transfer_delay_s) or transfer_delay_s < 0:
        raise ValueError("Transfer delay must be finite and non-negative")
    if not ratios or any(
        type(p) is not int or type(d) is not int or p <= 0 or d <= 0 for p, d in ratios
    ):
        raise ValueError("Ratios must contain positive P and D counts")
    ordered = sorted(requests, key=lambda request: request.created_time)
    prefill_times = {}
    p_engine = AsyncLLM.from_engine_args(p_args)
    try:
        for p_count in dict.fromkeys(p for p, _ in ratios):
            times = {}
            for node in range(p_count):
                partition = ordered[node::p_count]
                stats = await replay_partition(p_engine, partition)
                for request, record in zip(partition, stats):
                    times[request.request_id] = (record["last_event_time"], node)
            prefill_times[p_count] = times
    finally:
        p_engine.shutdown()

    summaries = []
    d_engine = AsyncLLM.from_engine_args(d_args)
    try:
        for p_count, d_count in ratios:
            times = prefill_times[p_count]
            events = sorted(ordered, key=lambda request: times[request.request_id][0])
            joined = []
            for node in range(d_count):
                sources = events[node::d_count]
                partition = [
                    TraceRequest(
                        request.request_id,
                        request.prompt_token_ids,
                        request.output_length,
                        times[request.request_id][0] + transfer_delay_s,
                    )
                    for request in sources
                ]
                stats = await replay_partition(d_engine, partition, decode=True)
                for request, record in zip(sources, stats):
                    prefill_finished_at, p_node = times[request.request_id]
                    joined.append(
                        {
                            "request_id": request.request_id,
                            "p_instance": p_node,
                            "d_instance": node,
                            "created_time": request.created_time,
                            "prefill_finished_at": prefill_finished_at,
                            "decode_created_time": record["created_time"],
                            "finished_at": record["last_event_time"],
                            "output_length": record["output_length"],
                            "decode_external_kv_tokens": record["ext_kv_hit_len"],
                            "decode_local_kv_tokens": record["local_kv_hit_len"],
                        }
                    )
            start = min(request.created_time for request in requests)
            finish = max(record["finished_at"] for record in joined)
            duration = finish - start
            output_tokens = sum(record["output_length"] for record in joined)
            summaries.append(
                {
                    "p_instances": p_count,
                    "d_instances": d_count,
                    "simulated_gpus": (
                        p_count * p_args.tensor_parallel_size
                        + d_count * d_args.tensor_parallel_size
                    ),
                    "num_requests": len(requests),
                    "output_tokens": output_tokens,
                    "start_time": start,
                    "finish_time": finish,
                    "duration_s": duration,
                    "requests_per_second": len(requests) / duration if duration else 0,
                    "output_tokens_per_second": output_tokens / duration
                    if duration
                    else 0,
                    "mean_e2e_s": sum(
                        record["finished_at"] - record["created_time"]
                        for record in joined
                    )
                    / len(joined),
                    "requests": joined,
                }
            )
    finally:
        d_engine.shutdown()
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--dataset", type=Path, help="JSONL tokenized trace; otherwise synthesize one"
    )
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--request-rate", type=float, default=100)
    parser.add_argument("--gpu-budget", type=int, default=4)
    parser.add_argument("--p-tp", type=int, default=1)
    parser.add_argument("--d-tp", type=int, default=1)
    parser.add_argument(
        "--ratios",
        nargs="+",
        help="P:D pairs, e.g. 1:3 2:2 3:1; default: all pairs fitting the GPU budget",
    )
    parser.add_argument(
        "--p-predictor",
        type=json.loads,
        default='{"name":"linear","base_ms":1,"per_token_ms":0.02}',
    )
    parser.add_argument(
        "--d-predictor", type=json.loads, default='{"name":"linear","base_ms":5}'
    )
    parser.add_argument("--num-gpu-blocks", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--transfer-delay-ms", type=float, default=0)
    parser.add_argument(
        "--output", type=Path, help="Write metrics and per-request timestamps as JSON"
    )
    args = parser.parse_args()
    for key in (
        "num_requests",
        "input_len",
        "output_len",
        "gpu_budget",
        "p_tp",
        "d_tp",
        "num_gpu_blocks",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.request_rate) or args.request_rate <= 0:
        parser.error("--request-rate must be finite and positive")
    if not math.isfinite(args.transfer_delay_ms) or args.transfer_delay_ms < 0:
        parser.error("--transfer-delay-ms must be finite and non-negative")
    if args.dataset:
        requests = [
            TraceRequest(**json.loads(line))
            for line in args.dataset.read_text().splitlines()
            if line.strip()
        ]
    else:
        requests = [
            TraceRequest(
                str(i),
                [
                    42 + (i >> (7 * position)) % 128
                    for position in range(args.input_len)
                ],
                args.output_len,
                i / args.request_rate,
            )
            for i in range(args.num_requests)
        ]
    if not requests:
        parser.error("The dataset must not be empty")
    if args.ratios:
        try:
            ratios = [tuple(map(int, ratio.split(":"))) for ratio in args.ratios]
            if any(len(pair) != 2 for pair in ratios):
                raise ValueError
        except ValueError:
            parser.error("--ratios must use P:D integer pairs")
    else:
        ratios = []
        for p in range(1, args.gpu_budget // args.p_tp + 1):
            remaining = args.gpu_budget - p * args.p_tp
            if remaining > 0 and remaining % args.d_tp == 0:
                ratios.append((p, remaining // args.d_tp))
    if not ratios or any(
        p <= 0 or d <= 0 or p * args.p_tp + d * args.d_tp != args.gpu_budget
        for p, d in ratios
    ):
        parser.error(
            "Each ratio must use positive P/D counts and exactly --gpu-budget GPUs"
        )
    max_length = max(
        len(request.prompt_token_ids) + request.output_length for request in requests
    )
    common = dict(
        model=args.model,
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=max_length,
        block_size=16,
        num_gpu_blocks_override=args.num_gpu_blocks,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    p_args = AsyncEngineArgs(
        **common,
        tensor_parallel_size=args.p_tp,
        additional_config={
            "simulator": {"mode": "NON_BLOCKING", "predictor": args.p_predictor},
        },
    )
    d_args = AsyncEngineArgs(
        **common,
        tensor_parallel_size=args.d_tp,
        kv_transfer_config=OFFLINE_CONNECTOR,
        additional_config={
            "simulator": {"mode": "NON_BLOCKING", "predictor": args.d_predictor},
        },
    )
    results = asyncio.run(
        sweep(requests, ratios, p_args, d_args, args.transfer_delay_ms / 1000)
    )
    print("P:D  GPUs  duration(s)  requests/s  output tokens/s")
    for result in results:
        print(
            f"{result['p_instances']}:{result['d_instances']}  "
            f"{result['simulated_gpus']}  "
            f"{result['duration_s']:.6f}  {result['requests_per_second']:.3f}  "
            f"{result['output_tokens_per_second']:.3f}"
        )
    best = max(results, key=lambda result: result["output_tokens_per_second"])
    print(
        f"Best allocation in this sweep: {best['p_instances']}P:{best['d_instances']}D"
    )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "results": results,
                    "best_ratio": [best["p_instances"], best["d_instances"]],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
