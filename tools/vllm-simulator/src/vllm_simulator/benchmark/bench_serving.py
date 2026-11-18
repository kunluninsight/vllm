# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the native serving benchmark against a dedicated NON_BLOCKING server."""

import json
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from vllm.benchmarks import serve
from vllm.benchmarks.lib import endpoint_request_func as endpoints
from vllm.utils.argparse_utils import FlexibleArgumentParser


class FrontendTiming(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )
    ingress_ms: NonNegativeFloat = 0
    ingress_us_per_token: NonNegativeFloat = 0
    first_chunk_ms: NonNegativeFloat = 0
    first_chunk_us_per_token: NonNegativeFloat = 0
    next_chunk_ms: NonNegativeFloat = 0

    def dispatch_at(self, arrival: float, prompt_len: int) -> float:
        return (
            arrival
            + self.ingress_ms / 1000
            + self.ingress_us_per_token * prompt_len / 1e6
        )

    def client_times(self, stats: dict) -> tuple[float, list[float], float]:
        engine = stats["ttft"]
        ttft = (
            engine
            + self.first_chunk_ms / 1000
            + self.first_chunk_us_per_token * stats["input_length"] / 1e6
        )
        received = ttft
        itls = []
        for gap in stats["itls"]:
            engine += gap
            # Tokens already ready during first-chunk delivery can arrive together.
            following = max(received, engine + self.next_chunk_ms / 1000)
            itls.append(following - received)
            received = following
        return ttft, itls, received


class BenchmarkClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    perf_counter = time

    async def sleep(self, seconds):
        self.now += seconds


async def simulation_benchmark(native_benchmark, frontend=None, **options):
    if frontend is None:
        frontend = FrontendTiming()
    backend = options["endpoint_type"]
    count = len(options["input_requests"])
    if backend not in ("openai", "vllm", "openai-chat"):
        raise ValueError(f"Unsupported simulation backend: {backend}")
    if options["probe_request_rate"]:
        raise ValueError("Omit probe traffic on the dedicated simulation server")
    cap = options["max_concurrency"]
    if cap and cap < count:
        raise ValueError("Simulation requires all requests to be in flight")
    clock = BenchmarkClock()
    native_arrivals = serve.get_request
    arrival_time = ContextVar("arrival_time", default=None)

    async def arrivals(samples, *traffic):
        # Only the native arrival generator sees a virtual sleep. Restore asyncio
        # before yielding to the benchmark, which launches real HTTP tasks.
        schedule = native_arrivals(samples, *traffic)
        while True:
            with patch.object(serve, "asyncio", clock):
                item = await anext(schedule, None)
            if item is None:
                break
            # The HTTP task inherits the context at creation, before the next arrival.
            arrival_time.set(clock.now)
            yield item

    async def request(request_func_input, session, pbar=None):
        req = request_func_input
        arrival = arrival_time.get()
        try:
            payload = {"model": req.model_name or req.model}
            if backend == "openai-chat":
                messages = req.chat_messages
                if messages is None:
                    messages = endpoints._get_chat_messages(req)
                payload.update(messages=messages, max_completion_tokens=req.output_len)
            else:
                payload.update(
                    prompt=req.prompt,
                    max_tokens=req.output_len,
                    logprobs=req.logprobs,
                    repetition_penalty=1.0,
                )
            endpoints._update_payload_common(payload, req)
            prompt = payload.get("prompt")
            if payload.get("n", 1) != 1 or (
                isinstance(prompt, list) and not all(isinstance(t, int) for t in prompt)
            ):
                raise ValueError(
                    "Simulation expects one prompt and one choice per request"
                )
            xargs = dict(payload.get("vllm_xargs") or {})
            if "simulation" in xargs:
                raise ValueError("The benchmark client sets vllm_xargs.simulation")
            if arrival is not None:
                xargs["simulation"] = json.dumps(
                    dict(
                        created_time=arrival,
                        dispatch_at=frontend.dispatch_at(arrival, req.prompt_len),
                        total_requests=count,
                        abort_round_on_cancel=True,
                    )
                )
                payload["vllm_xargs"] = xargs
            payload["stream"] = False
            payload.pop("stream_options", None)
            headers = endpoints._get_headers("application/json")
            endpoints._update_headers_common(headers, req)
            async with session.post(req.api_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error = f"HTTP {resp.status}: {await resp.text()}"
                    if arrival is None:
                        return endpoints.RequestFuncOutput(error=error)
                    raise RuntimeError(error)
                data = await resp.json()
            stats = (data.get("kv_transfer_params") or {}).get("simulation") or {}
            if stats.get("mode") != "NON_BLOCKING":
                raise ValueError("Expected a NON_BLOCKING simulator server")
            ttft, itls, latency = frontend.client_times(stats)
            if arrival is not None:
                clock.now = max(clock.now, stats["created_time"] + latency)
            (choice,) = data["choices"]
            return endpoints.RequestFuncOutput(
                success=True,
                generated_text=(
                    choice.get("message", {}).get("content") or choice.get("text") or ""
                ),
                prompt_len=data["usage"]["prompt_tokens"],
                output_tokens=data["usage"]["completion_tokens"],
                start_time=stats["created_time"],
                ttft=ttft,
                itl=itls,
                latency=latency,
            )
        except BaseException:
            # A disconnect releases peers waiting for the rest of the workload.
            if arrival is not None:
                await session.close()
            raise
        finally:
            if pbar is not None:
                pbar.update(1)

    with (
        patch.object(serve, "time", clock),
        patch.object(serve, "get_request", arrivals),
        patch.dict(serve.ASYNC_REQUEST_FUNCS, {backend: request}),
    ):
        return await native_benchmark(**options)


def main():
    parser = FlexibleArgumentParser(description=__doc__)
    serve.add_cli_args(parser)
    parser.add_argument(
        "--frontend-timing", type=Path, help="Calibrated serving frontend timing JSON"
    )
    args = parser.parse_args()
    try:
        frontend = (
            FrontendTiming.model_validate_json(args.frontend_timing.read_bytes())
            if args.frontend_timing
            else FrontendTiming()
        )
    except (OSError, ValueError) as error:
        parser.error(f"--frontend-timing: {error}")
    with patch.object(
        serve, "benchmark", partial(simulation_benchmark, serve.benchmark, frontend)
    ):
        serve.main(args)


if __name__ == "__main__":
    main()
