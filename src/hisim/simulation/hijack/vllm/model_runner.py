"""CPU stand-in for vLLM's GPUModelRunner and the steps it models.

The runner loads a latency predictor in place of model weights and, like
GPUModelRunner, coordinates every real or dummy step across DP ranks before
predicting its span from the DP-wide batch.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from hisim.simulation.hijack.vllm.dp import DPBatchCoordinator
from hisim.simulation.hijack.vllm.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from hisim.simulation.manager import ConfigManager
from hisim.time_predictor import ScheduleBatch, ScheduleRequest
from hisim.utils import get_logger

logger = get_logger()


@dataclass
class SimStep:
    """One modeled step; every DP rank derives the same span from one batch."""

    forward_latency: float
    sample_tokens_latency: float
    logprobs_latency: float
    requests_by_dp: list[list[tuple[int, int, int]]]
    dp_rank: int = 0
    dp_step_id: Optional[int] = None
    prediction_cpu_latency: float = 0.0

    @property
    def full_step_latency(self) -> float:
        return self.forward_latency + self.sample_tokens_latency + self.logprobs_latency

    @property
    def blocking_wait_latency(self) -> float:
        # Prediction already consumed wall time; DP coordination is separate.
        return max(0.0, abs(self.full_step_latency) - self.prediction_cpu_latency)

    @property
    def requests(self) -> list[list[int]]:
        # Stats keep the predictor's `(extend, past_kv)` request format.
        return [[e, p] for reqs in self.requests_by_dp for e, p, _ in reqs]

    def dp_fields(self) -> dict:
        if len(self.requests_by_dp) == 1:
            return {}
        return {
            "dp_rank": self.dp_rank,
            "dp_size": len(self.requests_by_dp),
            "dp_step_id": self.dp_step_id,
            "requests_scope": "dp",
            "requests_by_dp": [
                [[e, p] for e, p, _ in reqs] for reqs in self.requests_by_dp
            ],
        }


class SimModelRunner:
    """CPU stand-in for GPUModelRunner.

    Holds the native KV specs and DP coordination state; `load_model` loads
    the latency predictor that replaces the model forward.
    """

    def __init__(self, vllm_config, kv_spec=None):
        self.vllm_config = vllm_config
        self._kv_spec = kv_spec
        self._dp = DPBatchCoordinator(vllm_config)
        self.model_info = None
        self.scheduler_config = None
        self.predictor = None
        self.model_memory_usage = 0

    def get_kv_cache_spec(self):
        return self._kv_spec

    def load_model(self):
        """Resolve target model/parallel settings and build the predictor."""
        self.model_info = resolve_model_info(self.vllm_config.model_config)
        self.scheduler_config = resolve_scheduler_config(self.vllm_config)
        ConfigManager.set_model_info(self.model_info)
        # Applies simulator overrides (data types, predictor backend) in place.
        ConfigManager.set_scheduler_config(self.scheduler_config)
        self.predictor = ConfigManager.get_inference_time_predictor(
            self.model_info,
            ConfigManager.get_accelerator_info(),
            self.scheduler_config,
        )
        logger.info(
            "[vLLM Hijack] ModelRunner loaded %s", type(self.predictor).__name__
        )

    def run_step(
        self,
        requests: list[tuple[int, int, int]],
        num_context_tokens: int = 0,
        logprobs_tokens: int = 0,
        is_dummy: bool = False,
    ) -> SimStep:
        """Synchronize the batch across DP ranks, then predict its span."""
        requests_by_dp, step_id = self._dp.coordinate(
            requests, num_context_tokens, is_dummy=is_dummy
        )
        prediction_start = perf_counter()
        step = self._predict(requests_by_dp, logprobs_tokens, step_id)
        step.prediction_cpu_latency = perf_counter() - prediction_start
        return step

    def _predict(
        self,
        requests_by_dp: list[list[tuple[int, int, int]]],
        logprobs_tokens: int = 0,
        dp_step_id: int | None = None,
    ) -> SimStep:
        """Predict one step from the DP-wide batch (identical on every DP rank).

        `predictor.dp_batch_scope` selects the predictor input: "global"
        (default) concatenates all ranks like the DP collector's training
        rows; "max_rank" predicts each rank's local batch and keeps the
        slowest, for predictors that model attention DP themselves.
        """
        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("ModelRunner predictor is not loaded")
        predictor_config = ConfigManager._get_raw_config().get("predictor", {})
        scope = predictor_config.get("dp_batch_scope", "global")
        if hasattr(predictor, "predict_dp_infer_time"):
            if scope != "global":
                raise ValueError("DP-aware ML requires dp_batch_scope=global")
            forward = float(predictor.predict_dp_infer_time(requests_by_dp))
            sample = float(predictor.predict_dp_sample_tokens_time(requests_by_dp))
            if not any(requests_by_dp):
                forward = float(predictor_config.get("dp_idle_step_ms", 0.0)) / 1e3
            return SimStep(
                forward_latency=forward,
                sample_tokens_latency=sample,
                logprobs_latency=logprobs_tokens
                * getattr(predictor, "logprobs_cost_s_per_token", 0.0),
                requests_by_dp=requests_by_dp,
                dp_rank=self._dp.dp_rank,
                dp_step_id=dp_step_id,
            )
        if scope == "global":
            groups = [[req for reqs in requests_by_dp for req in reqs]]
        elif scope == "max_rank":
            groups = list(requests_by_dp)
        else:
            raise ValueError(f"Unknown predictor.dp_batch_scope: {scope}")

        forward, sample = None, 0.0
        for group in groups:
            if not group:
                continue
            batch = ScheduleBatch(
                reqs=[
                    ScheduleRequest(extend_length=e, past_kv_length=p)
                    for e, p, _ in group
                ]
            )
            candidate = (
                float(predictor.predict_infer_time(batch)),
                float(predictor.predict_sample_tokens_time(batch))
                if hasattr(predictor, "predict_sample_tokens_time")
                else 0.0,
            )
            if forward is None or sum(candidate) > forward + sample:
                forward, sample = candidate
        if forward is None:
            # Every rank runs a dummy batch, e.g. within the DP step horizon tail.
            forward = float(predictor_config.get("dp_idle_step_ms", 0.0)) / 1e3

        # Optional logprobs cost applies only to local tokens requesting logprobs.
        lp_cost = getattr(predictor, "logprobs_cost_s_per_token", 0.0)
        return SimStep(
            forward_latency=forward,
            sample_tokens_latency=sample,
            logprobs_latency=lp_cost * logprobs_tokens,
            requests_by_dp=requests_by_dp,
            dp_rank=self._dp.dp_rank,
            dp_step_id=dp_step_id,
        )

    def __getattr__(self, name):
        # Public fallback methods are no-ops; private attributes stay absent.
        if name.startswith("_"):
            raise AttributeError(name)
        return _noop


def _noop(*args, **kwargs):
    """Module-level no-op function (picklable)."""
    return None
