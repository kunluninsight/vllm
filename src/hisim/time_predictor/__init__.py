from typing import TYPE_CHECKING
from hisim.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from hisim.time_predictor.linear import LinearTimePredictor

if TYPE_CHECKING:
    from hisim.time_predictor.aiconfigurator import AIConfiguratorTimePredictor

__all__ = (
    "ScheduleRequest",
    "ScheduleBatch",
    "InferTimePredictor",
    "LinearTimePredictor",
    "AIConfiguratorTimePredictor",
)


def __getattr__(name):
    # Linear prediction and common batch types do not require AIConfigurator.
    if name == "AIConfiguratorTimePredictor":
        from hisim.time_predictor.aiconfigurator import AIConfiguratorTimePredictor

        return AIConfiguratorTimePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
