# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinate replay admission, round lifecycle and request statistics."""

import time
from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple

import torch

from vllm.outputs import RequestError
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus
from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.plugin.state import ReplayMetadata
from vllm_simulator.request_manager import create_dispatcher
from vllm_simulator.request_manager.dispatcher import PendingRequest
from vllm_simulator.request_manager.stats import RequestStats


class RankReplayState(NamedTuple):
    now: float
    next_arrival: float
    has_scheduled_requests: bool
    received_count: int
    expected_count: int
    active_count: int
    pending_count: int
    round_validated: bool
    failed: bool


class RequestManager:
    def __init__(
        self,
        config,
        *,
        clock,
        dp,
        scheduler,
        notify_aborted: Callable[[list[Request]], None],
    ):
        self.clock = clock
        self.dp = dp
        self.scheduler = scheduler
        self.notify_aborted = notify_aborted
        mode = SimulatorConfig.from_config(config).mode
        self.blocking = mode == "BLOCKING"
        self.dispatcher = create_dispatcher(config)
        self.stats = RequestStats(mode)
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
        transfer = request.kv_transfer_params or {}
        if transfer:
            from vllm_simulator.plugin.kv_transfer.offline import (
                SimulatorOfflineKVConnector,
                validate_import,
            )

            if isinstance(self.scheduler.connector, SimulatorOfflineKVConnector):
                validate_import(transfer, request.num_prompt_tokens)
            elif "offline_kv_tokens" in transfer:
                raise ValueError(
                    "offline_kv_tokens requires SimulatorOfflineKVConnector"
                )
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
        local = RankReplayState(
            self.clock.now,
            self.dispatcher.next_time() if unpaused else float("inf"),
            any(self.scheduler.get_request_counts()),
            self.received_count,
            self.expected_count or 0,
            len(self.active),
            self.pending_count,
            self.round_validated,
            self.failure is not None,
        )
        states = (
            [local]
            if self.dp.size == 1
            else [
                RankReplayState(*row.tolist())
                for row in self.dp.gather(torch.tensor(local, dtype=torch.float64))
            ]
        )
        if any(state.failed for state in states):
            messages = self.dp.gather_json(self.failure)
            return self._fail_round(next(message for message in messages if message))
        expected = {
            int(state.expected_count) for state in states if state.received_count > 0
        }
        if len(expected) > 1:
            return self._fail_round("DP replay requests must agree on total_requests")
        total = sum(int(state.received_count) for state in states)
        barrier = next(iter(expected), 0)
        if barrier and total > barrier:
            return self._fail_round("DP received more requests than total_requests")
        if barrier and total < barrier:
            return not unpaused
        if (
            self.dispatcher.sync_metadata
            and barrier
            and not all(state.round_validated for state in states)
        ):
            records = self.dp.gather_json(self.dispatcher.round_metadata())
            try:
                self.dispatcher.validate_round(
                    [item for rank in records for item in rank]
                )
            except (ValueError, TypeError) as error:
                return self._fail_round(str(error))
            self.round_validated = True
        self.clock.now = max(state.now for state in states)
        if not any(state.has_scheduled_requests for state in states):
            arrival = min(next_completion, min(state.next_arrival for state in states))
            if arrival != float("inf"):
                self.clock.now = max(self.clock.now, arrival)
        candidates = self.dispatcher.candidates(self.clock.now) if unpaused else []
        cap = self.dispatcher.max_concurrency
        width = (
            min(cap, max(int(state.pending_count) for state in states)) if cap else 0
        )
        if width:
            candidates = self._select_global_candidates(candidates, width, cap, states)
        for entry in candidates:
            item = self.dispatcher.take(entry.request_id)
            self.active.add(entry.request_id)
            self.stats.admit(entry.request_id, self.clock.now)
            self.scheduler.admit_deferred(item.request)
        return True

    def _select_global_candidates(self, candidates, width, cap, states):
        slots = max(0, cap - sum(int(state.active_count) for state in states))
        times = torch.full((width,), float("inf"), dtype=torch.float64)
        for i, entry in enumerate(candidates):
            times[i] = entry.dispatch_at
        selected = sorted(
            (float(value), rank, index)
            for rank, values in enumerate(self.dp.gather(times))
            for index, value in enumerate(values)
            if float(value) != float("inf")
        )[:slots]
        indices = {index for _, rank, index in selected if rank == self.dp.rank}
        return [entry for i, entry in enumerate(candidates) if i in indices]

    def finish_step(self, outputs, snapshot):
        for rid in self.stats.update(snapshot, outputs):
            self.active.discard(rid)
            self.dispatcher.complete(rid)
        if self.dispatcher.sync_metadata:
            events = self.dp.gather_json(self.dispatcher.drain_events())
            self.dispatcher.apply_events([item for rank in events for item in rank])
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
        counts = (outstanding, self.received_count, self.expected_count or 0)
        ranks = (
            [counts]
            if self.dp.size == 1
            else [
                row.tolist()
                for row in self.dp.gather(torch.tensor(counts, dtype=torch.int64))
            ]
        )
        outstanding_by_rank, received_by_rank, expected_by_rank = zip(*ranks)
        if any(outstanding_by_rank) or sum(received_by_rank) < max(expected_by_rank):
            return
        self._clear_round()
