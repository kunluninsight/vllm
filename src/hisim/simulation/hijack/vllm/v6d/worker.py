"""Disable vLLM's v6d worker payload channels while retaining readiness signals.

The scheduler-side managers keep native daemon connections. Worker handlers
remain None; the simulated controller owns hybrid transfer acknowledgements.
"""

import asyncio
import time

import torch

from hisim.env import Envs
from hisim.hook import BaseHook
from hisim.utils import get_logger

logger = get_logger()


class C_V6dObjectConnectorWorkerHook(BaseHook):
    """Register cache placeholders without CUDA pinning or SRPC data handlers."""

    HOOK_CLASS_NAME = "V6dObjectConnectorWorker"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        def noop_register_v6d_host_memory(self):
            """Skip CUDA host registration; simulation never dereferences KV payloads."""
            logger.info(
                "[V6D Hijack] _register_v6d_host_memory: skipped "
                "(CPU direct mmap access)"
            )

        target._register_v6d_host_memory = noop_register_v6d_host_memory

        def override_async_v6d_init_and_notify(self) -> None:
            """Notify scheduler that V6D is ready without CUDA/SRPC init."""
            self._device = torch.device("cpu")
            self.client = None
            try:
                # The ready-RPC handler exists only in async registration mode;
                # sync mode is already ready and must not receive this opcode.
                if not Envs.vllm_async_registration_enabled():
                    logger.info(
                        "[V6D Hijack] skip CPU v6d ready notification "
                        "(sync register mode, scheduler ready by default)"
                    )
                    return

                from vllm.v1.hybrid_connector import IoDoneReqs, hybridworker
                from vllm.v1.hybrid_connector.engine_proxy import get_hybrid_worker_loop
                from vllm.v1.hybrid_connector.utils import kill_me_if_exception
                from vllm.v1.hybrid_connector.v6d_object_backend import (
                    _V6D_READY_REQ,
                    _V6D_READY_RESP,
                )

                @kill_me_if_exception
                async def _send_v6d_ready():
                    last_log_time = 0.0
                    while True:
                        try:
                            await hybridworker().io_done_rpc(
                                IoDoneReqs(worker_tprank=self._rank_id, reqids=[]),
                                _V6D_READY_REQ,
                                _V6D_READY_RESP,
                            )
                            logger.info(
                                "[V6D Hijack] sent CPU v6d ready notification "
                                "for tprank=%s",
                                self._rank_id,
                            )
                            return
                        except Exception as e:
                            if isinstance(e, AssertionError):
                                raise
                            if time.time() - last_log_time > 10:
                                logger.warning(
                                    "[V6D Hijack] send CPU v6d ready failed "
                                    "for tprank=%s, retrying: %s",
                                    self._rank_id,
                                    e,
                                )
                                last_log_time = time.time()
                            await asyncio.sleep(0.5)

                asyncio.run_coroutine_threadsafe(
                    _send_v6d_ready(), get_hybrid_worker_loop()
                )
            except Exception:
                logger.exception(
                    "[V6D Hijack] failed to schedule CPU v6d ready notification"
                )

        def override_start_async_v6d_init(self):
            """Skip worker-side V6D RPC/SRPC data channel initialization."""
            self._load_handler = None
            self._store_handler = None
            override_async_v6d_init_and_notify(self)
            logger.info(
                "[V6D Hijack] _start_async_v6d_init: skipped worker "
                "V6D RPC/SRPC data channel initialization (CPU mode)"
            )

        target._async_v6d_init_and_notify = override_async_v6d_init_and_notify
        target._start_async_v6d_init = override_start_async_v6d_init

        def override_register_kv_caches(self, kv_caches):
            """Register KV cache metadata without starting SRPC data channels."""
            self.kv_caches = kv_caches
            self._start_async_v6d_init()
            logger.info(
                "[V6D Hijack] register_kv_caches: registered %d tensors "
                "and skipped worker-side V6D/SRPC transport (CPU)",
                len(kv_caches),
            )

        target.register_kv_caches = override_register_kv_caches

        async def _completed_v6d_task():
            return None

        def override_async_start_store_kv(self, metadata):
            # Satisfy the worker task interface; controller deadlines own completion.
            req_ids = set(metadata.reqs_to_store)
            logger.debug(
                "[V6D Hijack] async_start_store_kv: completed CPU no-op stores %s",
                sorted(req_ids),
            )
            return {
                req_id: asyncio.create_task(_completed_v6d_task()) for req_id in req_ids
            }

        target.async_start_store_kv = override_async_start_store_kv

        logger.info("[V6D Hijack] V6dObjectConnectorWorker hook installed")
