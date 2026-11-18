# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predict the completion time of a scheduled batch."""

import math

from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.plugin.state import MonotonicClock, VirtualClock
from vllm_simulator.time_predictor import (
    ScheduleBatch,
    create_predictor,
)


class TimedModelOutput(AsyncModelRunnerOutput):
    """Synchronize the predicted device work when the executor consumes its output."""

    def __init__(self, output, clock, deadline):
        self.output = output
        self.clock = clock
        self.deadline = deadline

    def get_output(self, timeout=None):
        self.clock.wait_until(self.deadline, timeout)
        return self.output

    def get_kv_connector_output(self):
        return self.output.kv_connector_output


class SimulatorModelRunner:
    def __init__(self, config):
        self.simulator_config = SimulatorConfig.from_config(config)
        self.blocking = self.simulator_config.mode == "BLOCKING"
        self.clock = MonotonicClock() if self.blocking else VirtualClock()
        self.predictor = create_predictor(
            self.simulator_config.predictor_config(config)
        )
        self.last_step_start = 0.0
        self.submitted_at = 0.0

    def forward(self, batch: ScheduleBatch) -> None:
        self.submitted_at = self.clock.now
        duration = self.predictor.predict((batch,))
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Predictor duration must be finite and non-negative")
        self.last_step_start = max(self.submitted_at, self.clock.device_available_at)
        self.clock.device_available_at = self.last_step_start + duration
