"""Run remote-PLE admission without the PLE service or CPU offload consumer.

HiSim stubs the worker, so no offload process can attach fetched rows or
acknowledge them, and native admission would never set ``ple_ready``.
Readiness is marked locally; SimulatedKVController times the fetch.
"""

import asyncio
import os
from collections import deque
from dataclasses import dataclass

from hisim.hook import BaseHook
from hisim.utils import get_logger

logger = get_logger()


@dataclass(frozen=True)
class _SimReadyPLE:
    """Covers the full prompt, as native ``can_schedule`` requires."""

    token: str
    start: int
    end: int
    results: tuple = ()


class _SimPLEMetrics:
    def snapshot(self) -> dict:
        try:
            from vllm.v1.metrics.ple import empty_ple_metrics
        except ImportError:
            return {}
        return empty_ple_metrics()


class _SimPLETransport:
    """Attributes HybridScheduler reads from the native transport."""

    def __init__(self, timeout: float):
        self.owner = None
        self.timeout = timeout
        self.metrics = _SimPLEMetrics()

    async def handle_rpc(self, reader, writer):
        # No offload consumer sends ATTACHED/CONSUMED in simulation.
        writer.close()


class C_PLEConnectorHook(BaseHook):
    """Replace PLE fetch, attach, and cleanup with local readiness markers."""

    HOOK_CLASS_NAME = "PLEConnector"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector.ple_backend"

    @classmethod
    def hook(cls, target):
        def init(
            self,
            *,
            loop,
            get_request,
            send_attach,
            send_cleanup,
            tp_size,
            on_cleanup=lambda: None,
            transport=None,
        ):
            self.loop, self.get_request = loop, get_request
            self.transport = transport or _SimPLETransport(
                float(os.getenv("VLLM_PLE_REMOTE_TIMEOUT", "120"))
            )
            self._send_attach, self._send_cleanup = send_attach, send_cleanup
            self._tp_size = tp_size
            self._on_cleanup = on_cleanup
            self._fetching = {}
            self._finished_tokens = deque()
            self._releases = set()
            logger.info("[PLE Hijack] PLEConnector uses simulated admission")

        async def prepare(self, request):
            if getattr(request, "ple_terminal", False):
                raise asyncio.CancelledError
            if request.prompt_embeds is not None or request.prompt_token_ids is None:
                raise ValueError("Remote PLE requires prompt token IDs")
            if getattr(request, "ple_ready", None) is not None:
                return
            request.ple_dispatched = False
            request.ple_ready = _SimReadyPLE(
                f"sim:{request.request_id}", 0, request.num_prompt_tokens
            )

        def bind_scheduled(self, output):
            # The stubbed worker never consumes bindings; keep the native shape.
            for req_id in output.num_scheduled_tokens:
                request = self.get_request(req_id)
                ready = request.ple_ready
                if ready is None:
                    raise RuntimeError("Remote PLE request bypassed admission")
                output.ple_bindings[req_id] = {"token": ready.token, "layers": {}}
                request.ple_dispatched = True
            output.ple_finished_tokens = []

        def finish(self, request):
            request.ple_terminal = True
            request.ple_ready = None

        target.__init__ = init
        target.prepare = prepare
        target.bind_scheduled = bind_scheduled
        target.finish = finish
        target.has_cleanup = property(lambda self: False)
        logger.info("[PLE Hijack] PLEConnector hook installed")
