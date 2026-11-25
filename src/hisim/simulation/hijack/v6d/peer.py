"""Keep native v6d cache accounting while bypassing payload copies.

Use non-reserved bulkstore memory and TCP-only SRPC with soft-fail startup.
Tiered reads retain native lookup/eviction and task-local hit/key logging.
Remote metadata probes remain native to preserve per-object sizes and reject
stale tracker entries.
"""

from __future__ import annotations

import contextvars
import logging
import os
import time

from hisim.env import Envs
from hisim.hook import BaseHook

logger = logging.getLogger("hisim")

# Avoid RDMA dependencies in the daemon process.
os.environ.setdefault("SRPC_STREAM_DISABLE_RDMA", "1")

# Attribute native hit counters to each asyncio read batch. Connector-local
# cache shortcuts do not reach these hooks and are not counted.

_HITSOURCE_COUNTERS = ("local", "p2p", "sharedfs", "tair_kvcm")

# None outside a tracked read; otherwise this task's hit counters.
_hit_bag: contextvars.ContextVar = contextvars.ContextVar(
    "v6d_sim_hitsource_bag", default=None
)


def _hitsource_enabled() -> bool:
    return Envs.v6d_hitsource_enabled()


def _keysource_enabled() -> bool:
    """Key-lineage logging: keys= suffix + CreateSource/EvictSource lines."""
    return Envs.v6d_keysource_enabled()


def _fmt_keys(object_keys) -> str:
    """Join nonempty object keys for lineage logging."""
    keys = [k for k in (object_keys or ()) if k]
    return ",".join(keys)


# Transfer stubs leave native metadata, capacity, and eviction paths intact.


def _infer_transfer_bytes(args, kwargs) -> int:
    for key in ("sizes", "nbytes", "size", "length"):
        val = kwargs.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
        if isinstance(val, (list, tuple)):
            return sum(int(x) for x in val if isinstance(x, (int, float)) and x > 0)
    total = 0
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                sz = getattr(item, "size", None) or getattr(item, "nbytes", None)
                if isinstance(sz, (int, float)) and sz > 0:
                    total += int(sz)
        else:
            sz = getattr(arg, "size", None) or getattr(arg, "nbytes", None)
            if isinstance(sz, (int, float)) and sz > 0:
                total += int(sz)
    return total


def _sim_load_data_noop(*args, **kwargs):
    """Skip synchronous remote payload copying, with optional P2P latency."""
    bps = Envs.v6d_p2p_bandwidth_bps()
    base_s = Envs.v6d_p2p_base_s()
    if bps > 0.0 or base_s > 0.0:
        nbytes = _infer_transfer_bytes(args, kwargs)
        if nbytes > 0:
            delay = base_s + (nbytes / bps if bps > 0.0 else 0.0)
            if delay > 0.0:
                time.sleep(delay)
    logger.debug("[v6d-sim] skip remote blob data transfer (no data plane)")
    return None


def _sim_async_load_data_noop(*args, **kwargs):
    return []


class C_VineyardPeerHook(BaseHook):
    """Disable bulkstore pre-reservation and tolerate SRPC initialization failure."""

    HOOK_CLASS_NAME = "VineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.vineyard.peer"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def patched_init(self, argc=0, argv=None, *args, **kwargs):

            import v6d.common.transfer as transfer

            _original_init_srpc = transfer.init_srpc_transfer

            def _init_srpc_soft_fail(*args, **kwargs):
                try:
                    return _original_init_srpc(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        "[v6d-sim] SRPC init failed (%s); daemon continues, "
                        "P2P meta probing will fail like a dead peer",
                        e,
                    )
                    return None

            transfer.init_srpc_transfer = _init_srpc_soft_fail

            if argv is not None:
                argv = [
                    "--reserve_memory=false" if a == "--reserve_memory=true" else a
                    for a in argv
                ]
            return original_init(self, argc, argv, *args, **kwargs)

        target.__init__ = patched_init
        logger.info(
            "[v6d-sim] C_VineyardPeerHook installed: reserve_memory=false, "
            "SRPC init soft-fail"
        )


class C_TieredVineyardPeerHook(BaseHook):
    """Wrap native reads/create/eviction for logging and install transfer stubs."""

    HOOK_CLASS_NAME = "TieredVineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.tiered_vineyard.peer"

    @classmethod
    def hook(cls, target):
        original_acquire_tiered_read = target._acquire_tiered_read

        async def wrapped_acquire_tiered_read(self, *args, **kwargs):
            if not _hitsource_enabled():
                return await original_acquire_tiered_read(self, *args, **kwargs)
            object_keys = args[0] if args else kwargs.get("object_keys") or ()
            request_id = kwargs.get("request_id")
            bag = dict.fromkeys(_HITSOURCE_COUNTERS, 0)
            token = _hit_bag.set(bag)
            error = None
            try:
                return await original_acquire_tiered_read(self, *args, **kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                _hit_bag.reset(token)
                queried = len(object_keys)
                if queried:
                    hits = sum(bag.values())
                    keys_suffix = (
                        " keys=" + _fmt_keys(object_keys)
                        if _keysource_enabled()
                        else ""
                    )
                    logger.info(
                        "[V6D HitSource] request_id=%s queried=%d local=%d p2p=%d "
                        "sharedfs=%d tair_kvcm=%d miss=%d%s%s",
                        request_id,
                        queried,
                        bag["local"],
                        bag["p2p"],
                        bag["sharedfs"],
                        bag["tair_kvcm"],
                        max(queried - hits, 0),
                        f" error={error}" if error else "",
                        keys_suffix,
                    )

        target._acquire_tiered_read = wrapped_acquire_tiered_read

        # Log requested CREATE keys, including existing objects merely touched.
        original_acquire = target.acquire

        async def wrapped_acquire(self, object_keys, scope, *args, **kwargs):
            if not _keysource_enabled() or "CREATE" not in str(scope):
                return await original_acquire(self, object_keys, scope, *args, **kwargs)
            request_id = kwargs.get("request_id")
            if request_id is None:
                request_id = next((a for a in args[3:] if isinstance(a, str)), None)
            try:
                return await original_acquire(self, object_keys, scope, *args, **kwargs)
            finally:
                keys = _fmt_keys(object_keys)
                if keys:
                    logger.info(
                        "[V6D CreateSource] request_id=%s n=%d keys=%s",
                        request_id,
                        len([k for k in (object_keys or ()) if k]),
                        keys,
                    )

        target.acquire = wrapped_acquire

        # Log evicted keys for cross-request cache lineage.
        original_evict_batch = target._evict_batch_from_vineyard

        async def wrapped_evict_batch(self, object_keys, *args, **kwargs):
            if not _keysource_enabled():
                return await original_evict_batch(self, object_keys, *args, **kwargs)
            keys = _fmt_keys(object_keys)
            try:
                return await original_evict_batch(self, object_keys, *args, **kwargs)
            finally:
                if keys:
                    logger.info(
                        "[V6D EvictSource] n=%d keys=%s",
                        len([k for k in (object_keys or ()) if k]),
                        keys,
                    )

        target._evict_batch_from_vineyard = wrapped_evict_batch

        # Keep native metadata probes: mixed KV groups have different sizes,
        # and tracker entries can outlive the remote objects they describe.
        import v6d.common.transfer as transfer

        transfer.load_data = _sim_load_data_noop
        transfer.async_load_data = _sim_async_load_data_noop
        logger.info(
            "[v6d-sim] C_TieredVineyardPeerHook installed: real "
            "_acquire_tiered_read + [V6D HitSource] stats + real SRPC meta "
            "probe (payload copies stubbed); key lineage (keys= suffix + "
            "CreateSource/EvictSource) %s",
            "on" if _keysource_enabled() else "off",
        )


class C_HitRateStatsHook(BaseHook):
    """Count native hit classifications in the current read's context bag."""

    HOOK_CLASS_NAME = "HitRateStats"
    HOOK_MODULE_NAME = "v6d.server.peers.tiered_vineyard.stats"

    @classmethod
    def hook(cls, target):
        def wrap(name, counter):
            original = getattr(target, name)

            def wrapped(self, size: int = 0):
                bag = _hit_bag.get()
                if bag is not None:
                    bag[counter] += 1
                return original(self, size)

            setattr(target, name, wrapped)

        wrap("record_local_vineyard_hit", "local")
        wrap("record_remote_vineyard_hit", "p2p")
        wrap("record_sharedfs_hit", "sharedfs")
        wrap("record_tair_kvcm_hit", "tair_kvcm")
        logger.info(
            "[v6d-sim] C_HitRateStatsHook installed: record_*_hit -> "
            "[V6D HitSource] bag"
        )
