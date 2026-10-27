"""Wrap hooked AsyncLLM for simulation benchmarks.

Install hooks before vLLM imports, forward request replay metadata, and read
statistics exported by engine profile hooks at round boundaries.
"""

import asyncio
import json
import os
import uuid

from hisim.dataset import GenericRequest
from hisim.env import Envs
from hisim.simulation.benchmark import BaseWorker
from hisim.simulation.hijack.startup import init_hook
from hisim.utils import get_logger

# Environment must be set before vllm import
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")
os.environ.setdefault("HISIM_ENABLE", "1")

init_hook()

from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

# The benchmark-facing EngineArgs alias must support AsyncLLM-specific fields.
EngineArgs = AsyncEngineArgs

logger = get_logger("hisim")


# Replace untouched engine defaults; keep async_scheduling user-controlled.
_SIMULATION_DEFAULTS = {
    "enforce_eager": True,
    "load_format": "dummy",
}


def _resolve_prompt(req: GenericRequest):
    """Resolve prompt from GenericRequest (text or token_ids)."""
    from vllm import TokensPrompt

    if req.prompt is not None:
        return req.prompt
    if req.token_ids is not None:
        return TokensPrompt(prompt_token_ids=req.token_ids)
    raise ValueError("Request must have either prompt or token_ids")


class VLLMWorker(BaseWorker):
    """Benchmark worker with simulation defaults for untouched AsyncEngineArgs fields."""

    def __init__(
        self,
        engine_args: EngineArgs,
        name: str = "vllm_worker0",
    ):
        super().__init__(name)

        # Preserve an explicit output path and share it with engine processes.
        self.output_dir = Envs.worker_output_dir(name)

        # Apply simulation defaults for fields still at their EngineArgs default
        for field_name, sim_default in _SIMULATION_DEFAULTS.items():
            current = getattr(engine_args, field_name)
            ea_default = getattr(AsyncEngineArgs, field_name, None)
            if current == ea_default:
                setattr(engine_args, field_name, sim_default)

        # AsyncLLM binds its output handler on first generation.
        self._llm = AsyncLLM.from_engine_args(engine_args)
        logger.info("[VLLMWorker] Initialized with model=%s", engine_args.model)

        # trigger_simulation alternates start/stop profile (round boundaries)
        self._profile_is_start = True

    # Async benchmark interface.

    async def async_generate(self, req: GenericRequest):
        """Stream a single request through the engine; returns the final output."""
        # Pass simulation metadata (created_time) via extra_args
        extra_args = None
        if req.custom_params:
            sim_meta = {}
            if "created_time" in req.custom_params:
                sim_meta["created_time"] = req.custom_params["created_time"]
            if "total_request" in req.custom_params:
                sim_meta["total_request"] = req.custom_params["total_request"]
            if sim_meta:
                extra_args = {"simulation": sim_meta}

        sp = SamplingParams(
            max_tokens=req.output_length, ignore_eos=True, extra_args=extra_args
        )
        prompt = _resolve_prompt(req)

        final_output = None
        async for output in self._llm.generate(prompt, sp, request_id=uuid.uuid4().hex):
            final_output = output
        return final_output

    async def trigger_simulation(self, output_dir: str | None = None):
        """Alternate profile calls to export/reset statistics, leaving caches intact."""
        if self._profile_is_start:
            await self._llm.start_profile()
        else:
            await self._llm.stop_profile()
        self._profile_is_start = not self._profile_is_start

    async def pause_generation(self):
        pass

    async def continue_generation(self):
        pass

    # Statistics exported by the engine's profile hook.

    def _load_jsonl(self, filename: str) -> list[dict]:
        data = []
        file_path = os.path.join(self.output_dir, filename)
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            logger.error(f"The statistics data({file_path}) does not exist.")
        return data

    def get_request_stats(self) -> list[dict]:
        return self._load_jsonl("request.jsonl")

    def get_iteration_stats(self) -> list[dict]:
        return self._load_jsonl("iteration.jsonl")

    def reset_stats(self):
        """Statistics reset through trigger_simulation, not in this wrapper."""
        pass

    # Synchronous convenience interface.

    def generate(self, req: GenericRequest):
        """Generate output for a single request (sync).

        Only safe as the first generate entrypoint on this worker: AsyncLLM
        binds its output handler to the loop of the first generate() call.
        """
        return asyncio.run(self.async_generate(req))

    def flush_cache(self):
        """Not yet supported for vLLM simulation."""
        pass

    def shutdown(self):
        """Shutdown the vLLM engine and its background process."""
        self._llm.shutdown()
        logger.info("[VLLMWorker] Shutting down.")
