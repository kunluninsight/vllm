# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain native scheduling while accounting for deferred engine requests."""

from collections.abc import Iterable

from vllm.outputs import RequestError
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request, RequestStatus
from vllm_simulator.plugin.kv_transfer.disaggregated import SimulatorPDConnector
from vllm_simulator.plugin.platform import SimulatorPlatform
from vllm_simulator.request_manager.manager import RequestManager


class SimulatorScheduler(Scheduler):
    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
        super().__init__(
            SimulatorPlatform.get_scheduler_config(vllm_config, kv_cache_config),
            kv_cache_config,
            *args,
            **kwargs,
        )
        self.request_manager: RequestManager | None = None

    def add_request(self, request: Request) -> None:
        if (
            self.request_manager is None
            or request.abort_immediately
            or self.request_manager.receive(request)
        ):
            super().add_request(request)

    def admit_deferred(self, request: Request) -> None:
        super().add_request(request)

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        if isinstance(self.connector, SimulatorPDConnector):
            for batch in outputs.values():
                for output in batch.outputs:
                    message = self.connector.publish_errors.pop(output.request_id, None)
                    if message is not None:
                        output.new_token_ids = []
                        output.finish_reason = FinishReason.ERROR
                        output.error = RequestError(
                            code="kv_transfer_error", message=message
                        )
        return outputs

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        if (
            self.request_manager is None
            or finished_status != RequestStatus.FINISHED_ABORTED
        ):
            return super().finish_requests(request_ids, finished_status)
        if request_ids is not None:
            request_ids = (
                [request_ids] if isinstance(request_ids, str) else list(request_ids)
            )
        finished = super().finish_requests(request_ids, finished_status)
        return finished + self.request_manager.abort(request_ids)

    def get_num_unfinished_requests(self) -> int:
        count = super().get_num_unfinished_requests()
        if self.request_manager is not None:
            if self.pause_state == PauseState.UNPAUSED:
                count += self.request_manager.pending_count
            count += int(self.request_manager.has_pending_errors)
        if self.connector is not None and self.connector.has_pending_push_work():
            count += 1
        return count


class SimulatorAsyncScheduler(SimulatorScheduler, AsyncScheduler):
    """Replay admission with native in-flight token and block accounting."""
