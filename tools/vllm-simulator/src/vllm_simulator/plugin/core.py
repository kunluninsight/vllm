# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admit replay requests and account for completed native engine steps."""

from dataclasses import replace
from unittest.mock import patch

from vllm.v1.engine import core
from vllm.v1.engine.core import EngineCoreProc
from vllm_simulator.request_manager.manager import RequestManager


class SimulatorEngineCoreProc(EngineCoreProc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        executor = self.model_executor
        self.scheduler.request_manager = RequestManager(
            self.vllm_config,
            clock=executor.clock,
            scheduler=self.scheduler,
            notify_aborted=self._send_abort_outputs,
        )

    def _wait_for_output(self, future):
        if self.batch_queue is not None:
            self._retired_step = future.snapshot
        return super()._wait_for_output(future)

    def step_with_batch_queue(self):
        executor = self.model_executor
        manager = self.scheduler.request_manager
        self._retired_step = None
        # Native scheduling can queue empty cleanup outputs between arrivals.
        # Only pending device work bounds the virtual clock's idle advance.
        deadline = next(
            (
                future.snapshot.finished_at
                for future, _, _ in reversed(self.batch_queue)
                if future.snapshot.scheduled_ids
            ),
            float("inf"),
        )
        ready = manager.prepare_step(next_completion=deadline)
        outputs, executed = (
            super().step_with_batch_queue()
            if ready or self.batch_queue
            else ({}, False)
        )
        if self._retired_step is None and not manager.has_pending_errors:
            return outputs, executed
        snapshot = self._retired_step or executor.step_snapshot()
        if manager.blocking:
            snapshot = replace(snapshot, finished_at=executor.clock.now)
        return manager.finish_step(outputs or {}, snapshot), executed

    def step(self):
        executor = self.model_executor
        manager = self.scheduler.request_manager
        executor.begin_step()
        outputs, executed = super().step() if manager.prepare_step() else ({}, False)
        return manager.finish_step(outputs, executor.step_snapshot()), executed


_native_launcher = EngineCoreProc.run_engine_core


def run_simulator_core(*args, **kwargs):
    with patch.object(core, "EngineCoreProc", SimulatorEngineCoreProc):
        _native_launcher(*args, **kwargs)


def install_simulator_launcher():
    # A module-level process target also works with multiprocessing spawn.
    EngineCoreProc.run_engine_core = staticmethod(run_simulator_core)
