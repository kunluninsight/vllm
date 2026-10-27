import pytest

from hisim.simulation.hijack.vllm.utils import resolve_model_info
from hisim.simulation.types import (
    SchedulerConfig,
)
from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo
from hisim.time_predictor import (
    LinearTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)


def test_time_predictor():
    pytest.importorskip("aiconfigurator")
    vllm = pytest.importorskip("vllm")
    from vllm.config import ModelConfig
    from hisim.time_predictor import AIConfiguratorTimePredictor

    model: ModelInfo = resolve_model_info(ModelConfig(model="Qwen/Qwen3-8B"))
    hw = AcceleratorInfo(
        name="a100_sxm",
        vendor="NVIDIA",
        hbm_capacity_gb=80,
        hbm_bandwidth_gb=2039,
        inter_node_bandwidth_gb=25,
        intra_node_bandwidth_gb=300,
    )
    config = SchedulerConfig(backend_name="vllm", backend_version=vllm.__version__)
    for clz in [
        AIConfiguratorTimePredictor,
    ]:
        predictor = clz(model, hw, config)

        # Prefill
        reqs = [
            ScheduleRequest(512, 512),
            ScheduleRequest(1024, 0),
            ScheduleRequest(512, 0),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0

        # Decode
        reqs = [
            ScheduleRequest(1, 1024),
            ScheduleRequest(1, 1024),
            ScheduleRequest(1, 1024),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0


if __name__ == "__main__":
    test_time_predictor()
