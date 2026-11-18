# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU worker calls and virtual batch completion times."""

from concurrent.futures import Future
from dataclasses import dataclass

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.executor.abstract import Executor
from vllm_simulator.plugin.model_runner import TimedModelOutput


@dataclass(frozen=True)
class StepSnapshot:
    started_at: float
    finished_at: float
    scheduled_ids: tuple[str, ...]
    dp_rank: int | None
    batch_tokens: tuple[int, ...]


class BatchFuture(Future):
    """A synthetic model output and its virtual completion deadline."""

    def __init__(self, output, snapshot):
        super().__init__()
        self.output = output
        self.snapshot = snapshot
        if not isinstance(output, TimedModelOutput):
            self.set_result(output)

    def result(self, timeout=None):
        if not self.done():
            try:
                self.set_result(self.output.get_output(timeout))
            except TimeoutError:
                raise
            except Exception as error:
                self.set_exception(error)
        return super().result(timeout)


class SimulatorExecutor(Executor):
    def _init_executor(self) -> None:
        worker_cls = resolve_obj_by_qualname(self.parallel_config.worker_cls)
        self.worker = worker_cls(self.vllm_config, 0, 0, "", is_driver_worker=True)
        self.worker.init_device()
        self.worker.load_model()
        self.clock = self.worker.model_runner.clock
        topology = self.worker.model_runner.simulator_config.original_parallel_config
        self.dp_index = (
            self.parallel_config.data_parallel_index
            if topology.data_parallel_size > 1
            else None
        )
        self.sample_future = None
        self.last_scheduled_ids: tuple[str, ...] = ()

    def initialize_from_config(self, configs) -> None:
        self.worker.initialize_from_config(configs[0])

    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block=False
    ):
        def call():
            if isinstance(method, str):
                return getattr(self.worker, method)(*args, **(kwargs or {}))
            return method(self.worker, *args, **(kwargs or {}))

        if not non_block:
            return [call()]
        future = Future()
        try:
            future.set_result([call()])
        except Exception as error:
            future.set_exception(error)
        return future

    @property
    def dp(self):
        return self.worker.model_runner.dp

    def bind_dp_group(self, group) -> None:
        self.worker.model_runner.bind_dp_group(group)

    def begin_step(self) -> None:
        self.last_scheduled_ids = ()

    def step_snapshot(self) -> StepSnapshot:
        runner = self.worker.model_runner
        return StepSnapshot(
            started_at=runner.last_step_start,
            finished_at=self.clock.now,
            scheduled_ids=self.last_scheduled_ids,
            dp_rank=self.dp_index,
            batch_tokens=tuple(batch.num_tokens for batch in runner.last_batches),
        )

    def execute_model(self, scheduler_output, non_block=False):
        self.last_scheduled_ids = tuple(scheduler_output.num_scheduled_tokens)
        submitted_at = self.clock.now
        output = self.worker.execute_model(scheduler_output)
        snapshot = StepSnapshot(
            submitted_at,
            self.clock.device_available_at if self.last_scheduled_ids else submitted_at,
            self.last_scheduled_ids,
            self.dp_index,
            (scheduler_output.total_num_scheduled_tokens,),
        )
        future = BatchFuture(output, snapshot)
        if self.scheduler_config.async_scheduling and self.last_scheduled_ids:
            self.sample_future = future
        return future if non_block else future.result()

    def sample_tokens(self, grammar_output, non_block=False):
        # The synthetic worker produces the sampled IDs with its forward.
        future, self.sample_future = self.sample_future, None
        if future is None:
            return super().sample_tokens(grammar_output, non_block)
        return future if non_block else future.result()

    def execute_dummy_batch(self) -> None:
        self.last_scheduled_ids = ()
        self.worker.execute_dummy_batch()
        self.clock.wait_until(self.clock.device_available_at)

    def check_health(self) -> None:
        pass

    def shutdown(self) -> None:
        self.worker.shutdown()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True
