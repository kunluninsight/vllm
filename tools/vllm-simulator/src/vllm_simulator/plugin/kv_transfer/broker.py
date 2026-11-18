# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared real-time metadata transport for independently served P/D instances."""

import argparse
import heapq
import json
import math
import threading
import time
import uuid
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm_simulator.plugin.kv_transfer.protocol import WireKVLayout


@dataclass
class Transfer:
    token: str
    source: str
    source_request: str
    layout: WireKVLayout
    prompt_hash: str
    num_tokens: int
    blocks: tuple[int, ...]
    created_at: float
    deadline: float
    destination: str | None = None
    destination_request: str | None = None
    sending_done: bool = False
    receiving_done: bool = False
    terminal: bool = False
    stats: dict | None = None
    allocated_at: float = 0
    started_at: float = 0
    sent_at: float = 0
    copy_started_at: float = 0
    ready_at: float = 0
    copy_s: float = 0


class TransferBroker:
    """Own FIFO bandwidth, per-destination copies, leases and acknowledged events."""

    def __init__(
        self,
        bandwidth_bytes_per_second,
        latency_s=0,
        reshard_bandwidth_bytes_per_second=None,
        lease_s=60,
    ):
        for value in (bandwidth_bytes_per_second, lease_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Bandwidth and lease must be finite and positive")
        if not math.isfinite(latency_s) or latency_s < 0:
            raise ValueError("Latency must be finite and non-negative")
        if reshard_bandwidth_bytes_per_second is not None and (
            not math.isfinite(reshard_bandwidth_bytes_per_second)
            or reshard_bandwidth_bytes_per_second <= 0
        ):
            raise ValueError("Reshard bandwidth must be finite and positive")
        self.bandwidth = bandwidth_bytes_per_second
        self.latency = latency_s
        self.reshard_bandwidth = reshard_bandwidth_bytes_per_second
        self.lease = lease_s
        self.broker_id = uuid.uuid4().hex
        self.condition = threading.Condition()
        self.transfers: dict[str, Transfer] = {}
        self._keys = {}
        self._owned = defaultdict(set)
        self._events = defaultdict(list)
        self._sequence = defaultdict(int)
        self._deadlines = []
        self._link_available = 0.0
        self._copy_available = {}

    def _event(self, job, kind, failed=False):
        if kind == "send":
            if job.sending_done:
                return
            job.sending_done = True
            owner, request_id = job.source, job.source_request
        else:
            if job.receiving_done or job.destination is None:
                return
            job.receiving_done = True
            owner, request_id = job.destination, job.destination_request
        self._sequence[owner] += 1
        self._events[owner].append(
            {
                "seq": self._sequence[owner],
                "kind": kind,
                "request_id": request_id,
                "transfer_id": job.token,
                "failed": failed,
                "stats": job.stats or {},
            }
        )
        self.condition.notify_all()

    def _finish(self, job, now, failed=False):
        if job.terminal:
            return
        if failed:
            self._event(job, "send", True)
            self._event(job, "recv", True)
        job.terminal = True
        job.deadline = now + self.lease
        heapq.heappush(self._deadlines, (job.deadline, "gc", job.token))

    def _tick(self, now):
        while self._deadlines and self._deadlines[0][0] <= now:
            when, kind, token = heapq.heappop(self._deadlines)
            job = self.transfers.get(token)
            if job is None:
                continue
            if kind == "gc":
                self.transfers.pop(token)
                self._keys.pop((job.source, job.source_request), None)
                for owner in (job.source, job.destination):
                    if owner is not None:
                        self._owned[owner].discard(token)
                        if not self._owned[owner]:
                            self._owned.pop(owner, None)
                continue
            if job.terminal:
                continue
            if kind == "expire" and when == job.deadline:
                self._finish(job, now, True)
            elif kind == "send":
                if when == job.sent_at:
                    self._event(job, "send")
            elif kind == "recv" and when == job.ready_at:
                self._event(job, "send")
                self._event(job, "recv")
                self._finish(job, now)

    def _reschedule(self, now):
        """Return cancelled FIFO/copy reservations without interrupting other work."""
        active = [
            j
            for j in self.transfers.values()
            if not j.terminal and j.destination is not None
        ]
        available = now
        for job in sorted(active, key=lambda j: j.started_at):
            if job.sending_done:
                continue
            if job.started_at > now:
                old_sent = job.sent_at
                job.started_at = available
                job.sent_at = available + job.stats["kv_transfer_s"]
                job.stats["kv_link_queue_s"] = available - job.allocated_at
                if job.sent_at != old_sent:
                    heapq.heappush(self._deadlines, (job.sent_at, "send", job.token))
            available = job.sent_at
        self._link_available = available
        copies = {}
        for job in sorted(active, key=lambda j: j.copy_started_at):
            old_ready = job.ready_at
            if job.copy_s:
                if job.copy_started_at > now:
                    job.copy_started_at = max(
                        job.sent_at, copies.get(job.destination, now)
                    )
                job.ready_at = job.copy_started_at + job.copy_s
                copies[job.destination] = job.ready_at
            else:
                job.copy_started_at = job.ready_at = job.sent_at
            job.stats["kv_reshard_queue_s"] = job.copy_started_at - job.sent_at
            if job.ready_at == old_ready:
                continue
            job.deadline = job.ready_at + self.lease
            heapq.heappush(self._deadlines, (job.ready_at, "recv", job.token))
            heapq.heappush(self._deadlines, (job.deadline, "expire", job.token))
        self._copy_available = copies

    def tick(self):
        with self.condition:
            self._tick(time.monotonic())

    def _lookup(self, token):
        job = self.transfers.get(token)
        if job is None or job.terminal:
            raise ValueError("KV handoff is unknown, expired or already completed")
        return job

    def publish(self, owner, request_id, layout, prompt_hash, num_tokens, blocks):
        with self.condition:
            now = time.monotonic()
            self._tick(now)
            key = owner, request_id
            if key in self._keys:
                return {"transfer_id": self._keys[key]}
            layout = WireKVLayout(**layout)
            if (
                layout.instance != owner
                or num_tokens <= 0
                or len(blocks) < math.ceil(num_tokens / layout.block_size)
            ):
                raise ValueError("Invalid P layout or source block reservation")
            token = f"{self.broker_id}:{uuid.uuid4().hex}"
            job = Transfer(
                token,
                owner,
                request_id,
                layout,
                prompt_hash,
                num_tokens,
                tuple(blocks),
                now,
                now + self.lease,
            )
            self.transfers[token] = job
            self._keys[key] = token
            self._owned[owner].add(token)
            heapq.heappush(self._deadlines, (job.deadline, "expire", token))
            self.condition.notify_all()
            return {"transfer_id": token}

    def lookup(self, transfer_id=None, source=None, source_request=None):
        with self.condition:
            self._tick(time.monotonic())
            if transfer_id is None:
                transfer_id = self._keys.get((source, source_request))
            job = self._lookup(transfer_id)
            return {
                "transfer_id": transfer_id,
                "num_tokens": job.num_tokens,
                "prompt_hash": job.prompt_hash,
                "source": job.source,
                "source_request": job.source_request,
                "stats": job.stats,
            }

    def receive(
        self,
        owner,
        request_id,
        transfer_id,
        layout,
        prompt_hash,
        local_tokens,
        blocks,
        *,
        source=None,
        source_request=None,
        num_tokens=None,
        destination_wait_s=None,
    ):
        with self.condition:
            now = time.monotonic()
            self._tick(now)
            if transfer_id is None:
                transfer_id = self._keys.get((source, source_request))
            job = self._lookup(transfer_id)
            if (
                (source is not None and source != job.source)
                or (source_request is not None and source_request != job.source_request)
                or (num_tokens is not None and num_tokens != job.num_tokens)
                or prompt_hash != job.prompt_hash
            ):
                raise ValueError("KV handoff does not match the request prompt/source")
            if job.destination is not None:
                if (job.destination, job.destination_request) == (owner, request_id):
                    return {**job.stats, "transfer_id": transfer_id}
                raise ValueError("KV handoff is already claimed by another request")
            destination = WireKVLayout(**layout)
            plan, reshard = job.layout.plan_to(destination)
            first = local_tokens // destination.block_size
            end = math.ceil(job.num_tokens / destination.block_size)
            if (
                destination.instance != owner
                or not 0 <= local_tokens < job.num_tokens
                or len(blocks) < end
            ):
                raise ValueError("Prompt or destination KV reservation does not match")
            shards = [replace(s, num_bytes=s.num_bytes * (end - first)) for s in plan]
            wire_bytes = sum(s.num_bytes for s in shards)
            destination_bytes = (
                (end - first) * destination.block_bytes * destination.tp_size
            )
            reshard_bytes = wire_bytes + destination_bytes if reshard else 0
            started = max(now, self._link_available)
            sent = started + self.latency + wire_bytes / self.bandwidth
            self._link_available = sent
            copy_start = sent
            copy_s = (
                reshard_bytes / self.reshard_bandwidth if self.reshard_bandwidth else 0
            )
            if copy_s:
                copy_start = max(sent, self._copy_available.get(owner, 0))
                self._copy_available[owner] = copy_start + copy_s
            ready = copy_start + copy_s
            job.destination, job.destination_request = owner, request_id
            job.allocated_at, job.started_at, job.sent_at = now, started, sent
            job.copy_started_at, job.copy_s, job.ready_at = copy_start, copy_s, ready
            self._owned[owner].add(transfer_id)
            job.stats = {
                "kv_transfer_bytes": wire_bytes,
                "kv_destination_bytes": destination_bytes,
                "kv_source_bytes": (end - first)
                * job.layout.block_bytes
                * job.layout.tp_size,
                "kv_transfer_shards": [asdict(s) for s in shards],
                "kv_offer_to_claim_s": now - job.created_at,
                "kv_destination_wait_s": destination_wait_s,
                "kv_link_queue_s": started - now,
                "kv_transfer_s": sent - started,
                "kv_reshard_bytes": reshard_bytes,
                "kv_reshard_s": copy_s,
                "kv_reshard_queue_s": copy_start - sent,
            }
            job.deadline = ready + self.lease
            for when, kind in (
                (sent, "send"),
                (ready, "recv"),
                (job.deadline, "expire"),
            ):
                heapq.heappush(self._deadlines, (when, kind, transfer_id))
            self.condition.notify_all()
            return {**job.stats, "transfer_id": transfer_id}

    def cancel(
        self, owner, transfer_id=None, *, request_id, source=None, source_request=None
    ):
        with self.condition:
            now = time.monotonic()
            self._tick(now)
            if transfer_id is None:
                transfer_id = self._keys.get((source, source_request))
            job = self.transfers.get(transfer_id)
            if job is not None and not job.terminal:
                if job.destination is not None and (owner, request_id) not in (
                    (job.source, job.source_request),
                    (job.destination, job.destination_request),
                ):
                    raise ValueError("KV handoff belongs to another request")
                self._finish(job, now, True)
                if job.destination is not None:
                    self._reschedule(now)
            return {}

    def poll(self, owner, ack=0, wait_s=0, broker_id=None):
        with self.condition:
            if broker_id is not None and broker_id != self.broker_id:
                ack = 0
                wait_s = 0
            deadline = time.monotonic() + min(max(wait_s, 0), 0.1)
            while True:
                now = time.monotonic()
                self._tick(now)
                self._events[owner] = [e for e in self._events[owner] if e["seq"] > ack]
                if self._events[owner] or now >= deadline:
                    return {
                        "broker_id": self.broker_id,
                        "events": self._events[owner].copy(),
                    }
                wake = min(
                    deadline, self._deadlines[0][0] if self._deadlines else deadline
                )
                self.condition.wait(max(0, wake - now))

    def close(self, owner):
        with self.condition:
            now = time.monotonic()
            self._tick(now)
            reschedule = False
            for token in list(self._owned.get(owner, ())):
                job = self.transfers[token]
                if job.terminal or (
                    job.source == owner
                    and job.sending_done
                    and job.destination != owner
                ):
                    continue
                self._finish(job, now, True)
                reschedule |= job.destination is not None
            if reschedule:
                self._reschedule(now)
            self._events.pop(owner, None)
            return {}

    def stats(self):
        with self.condition:
            self._tick(time.monotonic())
            jobs = [j for j in self.transfers.values() if not j.terminal]
            return {
                "pending_offers": sum(j.destination is None for j in jobs),
                "retained_sources": sum(not j.sending_done for j in jobs),
                "waiting_receivers": sum(
                    j.destination is not None and not j.receiving_done for j in jobs
                ),
            }


class BrokerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, broker):
        self.broker = broker
        super().__init__(address, BrokerHandler)

    def service_actions(self):
        self.broker.tick()


class BrokerHandler(BaseHTTPRequestHandler):
    def _reply(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"status": "ok"})
        elif self.path == "/stats":
            self._reply(200, self.server.broker.stats())
        else:
            self._reply(404, {"error": "Unknown broker endpoint"})

    def do_POST(self):
        methods = {
            name: getattr(self.server.broker, name)
            for name in ("publish", "lookup", "receive", "cancel", "poll", "close")
        }
        method = methods.get(self.path.removeprefix("/"))
        if method is None:
            self._reply(404, {"error": "Unknown broker endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 10 * 1024 * 1024:
                raise ValueError("Invalid broker request length")
            result = method(**json.loads(self.rfile.read(length)))
        except (ValueError, KeyError, TypeError) as error:
            self._reply(400, {"error": str(error)})
        else:
            self._reply(200, result)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Listen address")
    parser.add_argument("--port", type=int, default=9000, help="Listen port")
    parser.add_argument(
        "--bandwidth-bytes-per-second",
        type=float,
        required=True,
        help="Aggregate shared FIFO link bandwidth in bytes/s",
    )
    parser.add_argument(
        "--latency-ms", type=float, default=0, help="Per-transfer link latency"
    )
    parser.add_argument(
        "--reshard-bandwidth-bytes-per-second",
        type=float,
        help="Aggregate copy bandwidth per D instance; omitted means zero copy cost",
    )
    parser.add_argument(
        "--lease-seconds",
        type=float,
        default=60,
        help="Unclaimed handoff expiry and completion retention",
    )
    args = parser.parse_args()
    broker = TransferBroker(
        args.bandwidth_bytes_per_second,
        args.latency_ms / 1000,
        args.reshard_bandwidth_bytes_per_second,
        args.lease_seconds,
    )
    with (
        BrokerServer((args.host, args.port), broker) as server,
        suppress(KeyboardInterrupt),
    ):
        server.serve_forever(poll_interval=0.01)


if __name__ == "__main__":
    main()
