# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain native scheduling while accounting for deferred engine requests."""

from collections.abc import Iterable

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
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
        return count


class SimulatorAsyncScheduler(SimulatorScheduler, AsyncScheduler):
    """Replay admission with native in-flight token and block accounting."""
