# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather DP batch shapes and predict one shared forward duration."""

import math

import torch

from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.plugin.distributed import DPContext
from vllm_simulator.plugin.state import MonotonicClock, VirtualClock
from vllm_simulator.time_predictor import (
    ScheduleBatch,
    ScheduleRequest,
    create_predictor,
)


class TimedModelOutput(AsyncModelRunnerOutput):
    """Synchronize the predicted device work when the executor consumes its output."""

    def __init__(self, output, clock, deadline, finish_connector=None):
        self.output = output
        self.clock = clock
        self.deadline = deadline
        self.finish_connector = finish_connector

    def get_output(self, timeout=None):
        self.clock.wait_until(self.deadline, timeout)
        if self.finish_connector is not None:
            self.output.kv_connector_output = self.finish_connector()
            self.finish_connector = None
        return self.output

    def get_kv_connector_output(self):
        return self.output.kv_connector_output


class SimulatorModelRunner:
    def __init__(self, config):
        parallel = config.parallel_config
        self.dp = DPContext(parallel.data_parallel_rank, parallel.data_parallel_size)
        self.max_num_seqs = config.scheduler_config.max_num_seqs
        self.simulator_config = SimulatorConfig.from_config(config)
        self.blocking = self.simulator_config.mode == "BLOCKING"
        self.clock = MonotonicClock() if self.blocking else VirtualClock()
        self.predictor = (
            create_predictor(self.simulator_config.predictor_config(config))
            if self.rank == 0
            else None
        )
        self.last_batches: tuple[ScheduleBatch, ...] = ()
        self.last_step_start = 0.0
        self.submitted_at = 0.0

    def bind_dp_group(self, group) -> None:
        self.dp.group = group
        parallel = self.simulator_config.original_parallel_config
        layouts = self.dp.gather(
            torch.tensor(
                [
                    self.max_num_seqs,
                    int(self.blocking),
                    parallel.tensor_parallel_size,
                    parallel.pipeline_parallel_size,
                    parallel.prefill_context_parallel_size,
                    parallel.decode_context_parallel_size,
                    int(parallel.enable_expert_parallel),
                ]
            )
        )
        if len({tuple(layout[1:].tolist()) for layout in layouts}) != 1:
            raise ValueError(
                "All DP ranks must use the same simulation mode and topology"
            )
        self.max_num_seqs = max(int(layout[0]) for layout in layouts)

    @property
    def rank(self) -> int:
        return self.dp.rank

    @property
    def size(self) -> int:
        return self.dp.size

    def _gather_batches(self, batch: ScheduleBatch) -> tuple[ScheduleBatch, ...]:
        if self.size == 1:
            return (batch,)
        shape = torch.full((self.max_num_seqs + 1, 4), -1, dtype=torch.int64)
        shape[0, 0] = batch.batch_size
        for i, request in enumerate(batch.reqs, 1):
            shape[i] = torch.tensor(
                [
                    request.extend_length,
                    request.past_kv_length,
                    request.prompt_length,
                    request.num_prefill_tokens,
                ]
            )
        batches = []
        for rank, value in enumerate(self.dp.gather(shape)):
            rows = value[1 : 1 + int(value[0, 0])].tolist()
            batches.append(
                ScheduleBatch(reqs=[ScheduleRequest(*row) for row in rows], rank=rank)
            )
        return tuple(batches)

    def forward(self, batch: ScheduleBatch) -> None:
        self.submitted_at = self.clock.now
        self.last_batches = self._gather_batches(batch)
        timing = torch.zeros(3, dtype=torch.float64)
        error = None
        if self.rank == 0:
            assert self.predictor is not None
            if not self.blocking:
                timing[0] = self.clock.now
            try:
                duration = (
                    0.0
                    if not any(batch.num_tokens for batch in self.last_batches)
                    else self.predictor.predict(self.last_batches)
                )
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError(
                        "Predictor duration must be finite and non-negative"
                    )
                timing[1] = duration
            except Exception as exc:
                error = exc
                timing[2] = 1
        self.dp.sum(timing)
        if timing[2]:
            raise RuntimeError("Time prediction failed on DP rank 0") from error
        start = self.clock.now if self.blocking else float(timing[0])
        self.last_step_start = max(start, self.clock.device_available_at)
        self.clock.device_available_at = self.last_step_start + float(timing[1])

    def shutdown(self) -> None:
        self.dp.group = None
