"""Synchronize simulated batches across vLLM DP ranks inside the model runner.

Like GPUModelRunner, each real or dummy step calls the native
coordinate_batch_across_dp(). The DP utils hook extends its single _run_ar
all-reduce with each rank's numeric request tuples, using the collector's
layout, so every rank sees the same DP-wide batch without another collective.
"""

import inspect
import os
import threading
from functools import wraps

import torch

from hisim.env import Envs
from hisim.hook import BaseHook

_NATIVE_FIELDS = (
    "orig_num_tokens_per_ubatch",
    "padded_num_tokens_per_ubatch",
    "should_ubatch",
    "should_dp_pad",
    "cudagraph_mode",
    "num_reqs",
    "num_context_tokens",
    "is_prompt_batch_local",
    "is_prefill_only_local",
    "input_fits_in_drafter",
)
# Payload rows after the native ones: request count, step id, then tuples.
_HEADER_ROWS = 2
_INT32_MAX = 2**31 - 1
_CONTEXT = threading.local()


def _install_run_ar_hook(dp_utils) -> None:
    original = getattr(dp_utils, "_run_ar", None)
    if original is None:
        raise RuntimeError("HiSim DP simulation requires vLLM dp_utils._run_ar")
    if getattr(original, "_hisim_dp_payload", False):
        return
    signature = inspect.signature(original)
    if set(signature.parameters) != set(_NATIVE_FIELDS) | {
        "parallel_config",
        "device",
        "group",
    }:
        raise RuntimeError("Unsupported vLLM dp_utils._run_ar signature")
    num_native = len(_NATIVE_FIELDS)

    @wraps(original)
    def run_ar(*args, **kwargs):
        context = getattr(_CONTEXT, "current", None)
        # Only the simulated runner's coordination carries a payload.
        if context is None or context.get("payload") is not None:
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        rank, size = context["dp_rank"], context["dp_size"]
        parallel = values["parallel_config"]
        if (parallel.data_parallel_rank, parallel.data_parallel_size) != (rank, size):
            raise RuntimeError("HiSim DP coordination topology mismatch")
        requests = context["requests"]
        tensor = torch.zeros(
            num_native + _HEADER_ROWS + 3 * context["capacity"],
            size,
            device=values["device"],
            dtype=torch.int32,
        )
        for row, name in enumerate(_NATIVE_FIELDS):
            tensor[row, rank] = int(values[name])
        tensor[num_native, rank] = len(requests)
        tensor[num_native + 1, rank] = context["step_id"]
        if requests:
            packed = torch.tensor(requests, dtype=torch.int32).reshape(-1)
            start = num_native + _HEADER_ROWS
            tensor[start : start + packed.numel(), rank] = packed
        # Same group, operation, and collective count as the native _run_ar.
        torch.distributed.all_reduce(tensor, group=values["group"])
        context["payload"] = tensor[num_native:].cpu()
        return tensor[:num_native]

    run_ar._hisim_dp_payload = True
    dp_utils._run_ar = run_ar


class M_VLLMDPUtilsHook(BaseHook):
    """Carry per-rank request tuples on the native DP metadata all-reduce."""

    HOOK_MODULE_NAME = "vllm.v1.worker.dp_utils"

    @classmethod
    def hook(cls, target):
        _install_run_ar_hook(target)


class M_VLLMEngineUtilsHook(BaseHook):
    """Keep the inherited device mask when launching DP engine processes.

    The mock platform is not CUDA-alike, so vLLM would assign each local DP
    rank a device slice; simulated engines run on CPU and need no devices.
    """

    HOOK_MODULE_NAME = "vllm.v1.engine.utils"

    @classmethod
    def hook(cls, target):
        def get_device_indices(device_control_env_var, *args, **kwargs):
            return os.environ.get(device_control_env_var, "")

        target.get_device_indices = get_device_indices


def _decode_payload(payload, context) -> list[list[tuple[int, int, int]]]:
    counts = payload[0].tolist()
    steps = payload[1].tolist()
    if any(step != context["step_id"] for step in steps):
        raise RuntimeError(f"HiSim DP step counters are not aligned: {steps}")
    if any(count < 0 or count > context["capacity"] for count in counts):
        raise RuntimeError(f"HiSim DP request counts exceed capacity: {counts}")
    start = _HEADER_ROWS
    by_dp = [
        [
            tuple(req)
            for req in payload[start : start + 3 * count, rank]
            .reshape(count, 3)
            .tolist()
        ]
        for rank, count in enumerate(counts)
    ]
    if by_dp[context["dp_rank"]] != context["requests"]:
        raise RuntimeError("HiSim DP payload does not match the local batch")
    return by_dp


class DPBatchCoordinator:
    """Per-worker DP step state; mirrors the runner's native coordination call."""

    def __init__(self, vllm_config):
        parallel = vllm_config.parallel_config
        self.parallel_config = parallel
        self.dp_rank = parallel.data_parallel_rank
        self.dp_size = parallel.data_parallel_size
        speculative = vllm_config.speculative_config
        self.uniform_decode_query_len = 1 + (
            speculative.num_speculative_tokens if speculative else 0
        )
        # Sizes the all-reduce payload, so every rank must agree on it. A rank
        # schedules at most max_num_seqs requests per step.
        capacity = Envs.dp_max_requests()
        if capacity is None:
            capacity = vllm_config.scheduler_config.max_num_seqs
        if capacity < 1:
            raise ValueError(f"{Envs.DP_MAX_REQUESTS} must be positive")
        self.capacity = capacity
        self.step_id = 0

    def coordinate(
        self,
        requests: list[tuple[int, int, int]],
        num_context_tokens: int = 0,
        is_dummy: bool = False,
    ) -> tuple[list[list[tuple[int, int, int]]], int | None]:
        """Return (requests_by_dp, dp_step_id); DP1 skips coordination."""
        if self.dp_size == 1:
            return [list(requests)], None
        if len(requests) > self.capacity:
            raise ValueError(
                f"Batch of {len(requests)} requests exceeds the DP payload "
                f"capacity {self.capacity} (max_num_seqs or {Envs.DP_MAX_REQUESTS})"
            )
        if any(v < 0 or v > _INT32_MAX for req in requests for v in req):
            raise ValueError("HiSim DP request lengths must fit nonnegative int32")

        from vllm.v1.worker import dp_utils

        if not getattr(dp_utils._run_ar, "_hisim_dp_payload", False):
            raise RuntimeError("M_VLLMDPUtilsHook is not installed on vllm dp_utils")

        q = self.uniform_decode_query_len
        if is_dummy:
            # Matches Worker.execute_dummy_batch's native dummy shape.
            num_tokens = q
            is_prompt = is_prefill_only = False
            uniform_decode = True
        else:
            num_tokens = sum(req[0] for req in requests)
            is_prompt = any(req[0] != q for req in requests)
            is_prefill_only = all(req[0] != q for req in requests)
            uniform_decode = not is_prompt and num_tokens == len(requests) * q

        context = dict(
            requests=[tuple(req) for req in requests],
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            step_id=self.step_id,
            capacity=self.capacity,
            payload=None,
        )
        self.step_id += 1
        previous = getattr(_CONTEXT, "current", None)
        _CONTEXT.current = context
        try:
            # Simulation runs eager: no microbatching, DP padding, or graphs.
            dp_utils.coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                num_reqs=0 if is_dummy else len(requests),
                num_context_tokens=num_context_tokens,
                allow_microbatching=False,
                allow_dp_padding=False,
                parallel_config=self.parallel_config,
                num_tokens_padded=num_tokens,
                cudagraph_mode=0,
                is_prompt_batch_local=is_prompt,
                is_prefill_only_local=is_prefill_only,
                uniform_decode=uniform_decode,
                input_fits_in_drafter=True,
            )
        finally:
            _CONTEXT.current = previous
        if context["payload"] is None:
            raise RuntimeError("No native DP _run_ar observed during coordination")
        return _decode_payload(context["payload"], context), context["step_id"]
