from hisim.simulation.types import SchedulerConfig
from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo
from hisim.time_predictor.base import InferTimePredictor, ScheduleBatch
from hisim.utils import get_logger

logger = get_logger()


class LinearTimePredictor(InferTimePredictor):
    """Step latency = base_ms + per_token_ms * scheduled tokens.

    With per_token_ms=0 this is a constant per-step latency (debugging).
    """

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        base_ms: float = 0.0,
        per_token_ms: float = 0.0,
    ):
        super().__init__(model, hw, config)
        if base_ms < 0 or per_token_ms < 0:
            raise ValueError(
                "LinearTimePredictor base_ms and per_token_ms must be "
                f"non-negative, got base_ms={base_ms!r}, "
                f"per_token_ms={per_token_ms!r}"
            )
        self.base_s = float(base_ms) / 1e3
        self.per_token_s = float(per_token_ms) / 1e3
        logger.info(
            "LinearTimePredictor: base = %.6f s, per_token = %.9f s",
            self.base_s,
            self.per_token_s,
        )

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0
        num_tokens = sum(req.extend_length for req in batch.reqs)
        return self.base_s + self.per_token_s * num_tokens
