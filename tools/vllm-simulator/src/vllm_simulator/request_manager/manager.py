# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinate replay admission, round lifecycle and request statistics."""

import time
from collections import defaultdict
from collections.abc import Callable

from vllm.outputs import RequestError
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus
from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.plugin.state import ReplayMetadata
from vllm_simulator.request_manager import create_dispatcher
from vllm_simulator.request_manager.dispatcher import PendingRequest
from vllm_simulator.request_manager.stats import RequestStats


class RequestManager:
    def __init__(
        self,
        config,
        *,
        clock,
        scheduler,
        notify_aborted: Callable[[list[Request]], None],
    ):
        self.clock = clock
        self.scheduler = scheduler
        self.notify_aborted = notify_aborted
        self.blocking = SimulatorConfig.from_config(config).mode == "BLOCKING"
        self.dispatcher = create_dispatcher(config)
        self.stats = RequestStats()
        self.active = set()
        self.received_count = 0
        self.expected_count = None
        self.round_validated = False
        self.failure = None
        self._errors = defaultdict(list)

    @property
    def pending_count(self):
        return self.dispatcher.pending_count

    @property
    def has_pending_errors(self):
        return bool(self._errors) or self.failure is not None

    def _reject(self, request, message):
        self._errors[request.client_index].append(
            EngineCoreOutput(
                request.request_id,
                [],
                finish_reason=FinishReason.ERROR,
                error=RequestError(code="validation_error", message=message),
            )
        )

    def receive(self, request):
        try:
            return self._receive(request)
        except (ValueError, TypeError) as error:
            self._reject(request, str(error))
            if not self.blocking:
                self.failure = str(error)
            return False

    def _receive(self, request):
        if self.blocking:
            now = self.clock.now
            created = now - max(0.0, time.time() - request.arrival_time)
            item = PendingRequest(request, ReplayMetadata(created, now, None))
            self.dispatcher.add(item)
            self.stats.add(item)
            self.stats.admit(request.request_id, now)
            self.active.add(request.request_id)
            return True
        metadata = ReplayMetadata.from_extra_args(
            request.sampling_params.extra_args, self.clock.now
        )
        expected = metadata.total_requests
        if self.received_count and self.expected_count != expected:
            raise ValueError(
                "All requests in a replay round must agree on total_requests"
            )
        if expected is not None and self.received_count >= expected:
            raise ValueError("Received more requests than total_requests")
        item = PendingRequest(request, metadata)
        self.dispatcher.add(item)
        self.stats.add(item)
        self.expected_count = expected
        self.received_count += 1
        return False

    def _clear_round(self):
        self.dispatcher.reset()
        self.stats.reset()
        self.active.clear()
        self.received_count = 0
        self.expected_count = None
        self.round_validated = False
        self.failure = None
        if not self.blocking:
            self.clock.reset()

    def _fail_round(self, message):
        requests = self.dispatcher.cancel_pending()
        requests.extend(
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ERROR)
        )
        for request in requests:
            self._reject(request, message)
        self._clear_round()
        return False

    def prepare_step(self, *, next_completion=float("inf")):
        if self.blocking:
            return True
        unpaused = self.scheduler.pause_state == PauseState.UNPAUSED
        if self.failure:
            return self._fail_round(self.failure)
        if self.expected_count and self.received_count < self.expected_count:
            return not unpaused
        if (
            self.dispatcher.sync_metadata
            and self.expected_count
            and not self.round_validated
        ):
            try:
                self.dispatcher.validate_round(self.dispatcher.round_metadata())
            except (ValueError, TypeError) as error:
                return self._fail_round(str(error))
            self.round_validated = True
        if unpaused and not any(self.scheduler.get_request_counts()):
            arrival = min(next_completion, self.dispatcher.next_time())
            if arrival != float("inf"):
                self.clock.now = max(self.clock.now, arrival)
        candidates = self.dispatcher.candidates(self.clock.now) if unpaused else []
        cap = self.dispatcher.max_concurrency
        if cap:
            candidates = candidates[: max(0, cap - len(self.active))]
        for entry in candidates:
            item = self.dispatcher.take(entry.request_id)
            self.active.add(entry.request_id)
            self.stats.admit(entry.request_id, self.clock.now)
            self.scheduler.admit_deferred(item.request)
        return True

    def finish_step(self, outputs, snapshot):
        for rid in self.stats.update(snapshot, outputs):
            self.active.discard(rid)
            self.dispatcher.complete(rid)
        if self.dispatcher.sync_metadata:
            self.dispatcher.apply_events(self.dispatcher.drain_events())
        self._reset_if_drained()
        for client, errors in self._errors.items():
            batch = outputs.setdefault(client, EngineCoreOutputs())
            batch.outputs.extend(errors)
            if batch.finished_requests is None:
                batch.finished_requests = set()
            batch.finished_requests.update(output.request_id for output in errors)
        self._errors.clear()
        return outputs

    def abort(self, request_ids):
        if request_ids is None:
            deferred = self.dispatcher.cancel_pending()
            self._clear_round()
        else:
            deferred = []
            for rid in request_ids:
                if rid not in self.active:
                    try:
                        item = self.dispatcher.take(rid)
                    except KeyError:
                        continue
                    deferred.append(item.request)
                    if item.metadata.abort_round_on_cancel:
                        self.failure = "HTTP replay round cancelled"
                self.dispatcher.complete(rid)
                self.active.discard(rid)
                self.stats.discard(rid)
            # Native pause/shutdown notify bulk aborts. An explicit client abort
            # needs the IDs the native scheduler never admitted, for DP accounting.
            self.notify_aborted(deferred)
        return deferred

    def _reset_if_drained(self):
        if self.blocking:
            return
        outstanding = bool(self.pending_count) or any(
            self.scheduler.get_request_counts()
        )
        if not outstanding and self.received_count >= (self.expected_count or 0):
            self._clear_round()
