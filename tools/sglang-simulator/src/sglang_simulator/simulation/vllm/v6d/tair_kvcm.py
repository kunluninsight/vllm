"""v6d daemon-side Tair KVCM hook — keep KVCM control plane, stub MemPool I/O.

With ``--tair-kvcm-mode=report-event-and-cache-transfer`` the daemon creates a
``TairKVCMTransferRuntime`` wrapping ``kvcm_py_client.TransferClient``.  The
native client needs a local tair_mempool_server (TENT on CPU hosts); when it
cannot be created v6d silently stops reporting BLOCK_ADD / writing to KVCM.
Here the client is replaced by a no-op and the payload batches succeed
without moving bytes, so every KVCM control-plane call (location allocation,
event report, lookup, reclaim) still runs for real against the manager.
"""

import math
import os
import time

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()



def _tair_bandwidth_bps(env_name: str) -> float:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return 0.0
    try:
        gb_s = float(raw)
    except ValueError:
        raise ValueError(f"{env_name} must be a non-negative finite number") from None
    if not math.isfinite(gb_s) or gb_s < 0:
        raise ValueError(f"{env_name} must be a non-negative finite number")
    return gb_s * 1e9


def _tair_base_s(env_name: str | None) -> float:
    if not env_name:
        return 0.0
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(f"{env_name} must be a non-negative finite number") from None
    if not math.isfinite(val) or val < 0:
        raise ValueError(f"{env_name} must be a non-negative finite number")
    return val


def _sleep_for_bytes(sizes, env_name: str, base_env: str | None = None) -> None:
    bps = _tair_bandwidth_bps(env_name)
    base_s = _tair_base_s(base_env)
    if bps <= 0 and base_s <= 0:
        return
    total_bytes = sum(int(s) for s in sizes if s and s > 0)
    if total_bytes > 0:
        delay = base_s + (total_bytes / bps if bps > 0 else 0.0)
        if delay > 0:
            time.sleep(delay)


class _NoopTransferClient:
    """Stands in for kvcm_py_client.TransferClient; its I/O users are stubbed."""

    def close(self):
        pass


class C_TairKVCMTransferRuntimeHook(BaseHook):
    HOOK_CLASS_NAME = "TairKVCMTransferRuntime"
    HOOK_MODULE_NAME = "v6d.common.tair_kvcm.transfer"

    @classmethod
    def hook(cls, target):
        if target.__dict__.get("_sim_tair_kvcm_transfer_hook", False):
            return

        def create_transfer_client(self):
            return _NoopTransferClient()

        def batch_load_to_memory(self, uris, pointers, sizes):
            if not uris:
                return True
            if not (len(uris) == len(pointers) == len(sizes)):
                raise ValueError("uris, pointers, and sizes length mismatch")
            from v6d.common.tair_kvcm.transfer import _record_transfer_result

            _sleep_for_bytes(sizes, "HISIM_TAIR_KVCM_READ_BANDWIDTH_GB", "HISIM_TAIR_KVCM_READ_BASE_S")
            _record_transfer_result("read", True)
            return True

        def batch_save_from_memory(self, uris, pointers, sizes):
            if not uris:
                return []
            if not (len(uris) == len(pointers) == len(sizes)):
                raise ValueError("uris, pointers, and sizes length mismatch")
            from v6d.common.tair_kvcm.transfer import _record_transfer_result

            _sleep_for_bytes(sizes, "HISIM_TAIR_KVCM_WRITE_BANDWIDTH_GB", "HISIM_TAIR_KVCM_WRITE_BASE_S")
            _record_transfer_result("write", True)
            return [True] * len(uris)

        target._create_transfer_client = create_transfer_client
        target.batch_load_to_memory = batch_load_to_memory
        target.batch_save_from_memory = batch_save_from_memory
        target._sim_tair_kvcm_transfer_hook = True
        logger.info(
            "[v6d-sim] C_TairKVCMTransferRuntimeHook installed: native KVCM "
            "transfer client and MemPool payload I/O stubbed; control plane real"
        )
