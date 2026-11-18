# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy predictor selection, independent of plugin transport and hook state."""

from importlib import import_module

from vllm_simulator.plugin.config import SimulatorConfig

from .base import PerRankTimePredictor, ScheduleBatch, ScheduleRequest, TimePredictor

__all__ = [
    "PerRankTimePredictor",
    "ScheduleBatch",
    "ScheduleRequest",
    "TimePredictor",
    "create_predictor",
]


def create_predictor(config) -> TimePredictor:
    options = SimulatorConfig.from_config(config).predictor.copy()
    name = options.pop("name", "linear")
    if name == "linear":
        from .linear import LinearTimePredictor

        predictor = LinearTimePredictor(**options)
    elif name == "replay":
        from .replay import ReplayTimePredictor

        predictor = ReplayTimePredictor(**options)
    elif name == "ml":
        from .ml import MLTimePredictor

        predictor = MLTimePredictor(**options)
    elif name == "aisimulate":
        from .aisimulate import AISimulateTimePredictor

        predictor = AISimulateTimePredictor(config, **options)
    elif isinstance(name, str) and ":" in name:
        module, factory_name = name.split(":", 1)
        factory = getattr(import_module(module), factory_name)
        predictor = factory(config, **options)
    else:
        raise ValueError(f"Unknown time predictor: {name!r}")
    if not callable(getattr(predictor, "predict", None)):
        raise TypeError("A time predictor must provide predict(batches) -> seconds")
    return predictor
