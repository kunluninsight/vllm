# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-instance virtual-time benchmarks backed directly by AsyncLLM."""

import asyncio
import math
import random
import threading
import uuid
from concurrent.futures import Future
from copy import deepcopy

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncLLM
from vllm_simulator.benchmark.metrics import calc_metrics
from vllm_simulator.plugin.platform import SimulatorPlatform

EngineArgs = AsyncEngineArgs
__all__ = ["EngineArgs", "Benchmark"]


class Benchmark:
    def __init__(self, engine_args: AsyncEngineArgs):
        if not isinstance(current_platform, SimulatorPlatform):
            raise RuntimeError("Set VLLM_PLUGINS=simulator before importing vLLM")
        self._closed = False
        self._pending = {}
        self._running = False
        self.request_stats = []
        ready = Future()
        self._thread = threading.Thread(
            target=self._run_engine_loop,
            args=(engine_args, ready),
            name="benchmark-async-llm",
            daemon=True,
        )
        self._thread.start()
        try:
            ready.result()
        except BaseException:
            self._thread.join()
            raise

    def _run_engine_loop(self, engine_args, ready):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def initialize():
            self._llm = AsyncLLM.from_engine_args(engine_args)

        try:
            self._loop.run_until_complete(initialize())
            ready.set_result(None)
            self._loop.run_forever()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                raise
        finally:
            tasks = asyncio.all_tasks(self._loop)
            for task in tasks:
                task.cancel()
            if tasks:
                self._loop.run_until_complete(
                    asyncio.gather(*tasks, return_exceptions=True)
                )
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    @property
    def vllm_config(self):
        return self._llm.vllm_config

    def _submit(self, coroutine):
        if self._closed:
            coroutine.close()
            raise RuntimeError("Benchmark is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def benchmark(self, benchmark_config, requests):
        return self._submit(self._benchmark(benchmark_config, requests)).result()

    async def async_benchmark(self, benchmark_config, requests):
        return await asyncio.wrap_future(
            self._submit(self._benchmark(benchmark_config, requests))
        )

    async def _benchmark(self, config, requests):
        if self._running or self._pending:
            raise RuntimeError("Benchmark requires an idle engine")
        self._running = True
        self.request_stats = []
        tasks = []
        try:
            requests = deepcopy(list(requests))
            random_source = random.Random(config.seed)
            arrival = 0.0
            request_ids = set()
            for req in requests:
                rid = req.custom_params.setdefault("request_id", uuid.uuid4().hex)
                if not isinstance(rid, str) or not rid or rid in request_ids:
                    raise ValueError("Request ids must be non-empty and unique")
                request_ids.add(rid)
                req.custom_params["total_requests"] = len(requests)
                if config.ignore_request_timestamp:
                    req.custom_params["created_time"] = arrival
                    if math.isfinite(config.request_rate):
                        arrival += random_source.expovariate(config.request_rate)
                else:
                    req.custom_params.setdefault("created_time", 0.0)
            tasks = [asyncio.create_task(self._generate(req)) for req in requests]
            self.request_stats = await asyncio.gather(*tasks)
            return calc_metrics(self.request_stats)
        except BaseException:
            if tasks:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await self._cancel_replay_round()
            raise
        finally:
            self._running = False

    def get_request_stats(self):
        return self.request_stats

    async def async_generate(self, req):
        return await asyncio.wrap_future(self._submit(self._generate(req)))

    async def _generate(self, req):
        request_id = req.custom_params.setdefault("request_id", uuid.uuid4().hex)
        if (
            not isinstance(request_id, str)
            or not request_id
            or request_id in self._pending
        ):
            raise ValueError("Request ids must be non-empty and unique")
        options = deepcopy(req.sampling_kwargs)
        options.setdefault("max_tokens", req.output_length)
        options.setdefault("ignore_eos", True)
        options.setdefault("temperature", 0)
        extra_args = options.pop("extra_args", {})
        extra_args["simulation"] = {
            key: req.custom_params[key]
            for key in (
                "created_time",
                "dispatch_at",
                "total_requests",
                "conversation_id",
                "turn",
            )
            if key in req.custom_params
        }
        params = SamplingParams(extra_args=extra_args, **options)
        prompt = (
            req.prompt
            if req.prompt is not None
            else TokensPrompt(prompt_token_ids=req.token_ids)
        )
        self._pending[request_id] = asyncio.current_task()
        try:
            final_output = None
            async for output in self._llm.generate(
                prompt, params, request_id=request_id
            ):
                final_output = output
            stats = (
                (final_output.kv_transfer_params or {}).get("simulation")
                if final_output is not None
                else None
            )
            if stats is None:
                raise RuntimeError("Simulator engine returned no request stats")
            return {**stats, "rid": final_output.request_id}
        finally:
            self._pending.pop(request_id, None)

    def generate(self, req):
        return self._submit(self._generate(req)).result()

    def flush_cache(self):
        return self._submit(self._flush_cache()).result()

    async def async_flush_cache(self):
        await asyncio.wrap_future(self._submit(self._flush_cache()))

    async def _flush_cache(self):
        if self._running or self._pending:
            raise RuntimeError("Cache reset requires an idle, open benchmark")
        if not await self._llm.reset_prefix_cache(reset_connector=True):
            raise RuntimeError("Prefix cache reset failed")

    async def cancel_replay_round(self):
        await asyncio.wrap_future(self._submit(self._cancel_replay_round()))

    async def _cancel_replay_round(self):
        tasks = list(self._pending.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._llm.pause_generation(mode="abort", clear_cache=False)
        await self._llm.resume_generation()

    async def _shutdown(self):
        tasks = list(self._pending.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._llm.shutdown()

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
