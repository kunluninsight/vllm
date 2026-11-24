"""v6d daemon-side simulation hooks — run vineyardd on CPU without RDMA.

Applied by ``install_v6d_runtime_hooks()`` at vineyardd startup.  The C++
bulkstore is a sparse memfd mmap sized at the configured ``--vineyard-size``
(2M-aligned, ``reserve_memory=false``): blob pages are never written, so a
500G "cache" costs ~110MB RSS at boot plus ~10KB per object (measured: 500G
arena + 2000 objects x 132MB declared -> 130MB RSS, 0.0s boot).  Capacity
and eviction accounting are therefore the daemon's own — no virtual layer.

The read path is the REAL ``TieredVineyardPeer._acquire_tiered_read``
(reached via the connector). Native SRPC metadata probes preserve object
sizes and confirm that remote tracker entries still exist.
The payload copies (``transfer.load_data`` / ``async_load_data``, eager
strategies only) are always no-ops, and a ``[V6D HitSource]`` line per read
batch is logged (port of the production v6d_hitsource_patch; see
tmp.out/bugfix/hit_stats.sh).
With ``V6D_KEYSOURCE_LOG=1`` (default) each read line also carries its
object keys (`` keys=...`` suffix), and CREATE-scope acquires / eviction
batches log ``[V6D CreateSource]`` / ``[V6D EvictSource]`` lines — the
per-key lineage needed to trace multi-turn cache ancestry.
Only ``--peer=tiered_vineyard`` daemon mode is supported.
"""

from __future__ import annotations

import contextvars
import ipaddress
import math
import os
import sys
import time

from sglang_simulator.hook import BaseHook, install_class_hooks
from sglang_simulator.utils import get_logger

logger = get_logger()

# TCP-only SRPC engine: skips barex/RDMA and its libcuda/ibverbs deps.
os.environ.setdefault("SRPC_STREAM_DISABLE_RDMA", "1")

# ---------------------------------------------------------------------------
# [V6D HitSource] observability — sim port of the production daemon patch
# ---------------------------------------------------------------------------
#
# Byte-identical port of ``v6d_hitsource_patch`` (tmp.out/bugfix/hit_stats.sh):
# a contextvar counter bag is pushed around
# ``TieredVineyardPeer._acquire_tiered_read`` and bumped from
# ``HitRateStats.record_*_hit``; one line is logged per read batch:
#
#   [V6D HitSource] request_id=<rid> queried=N local=N p2p=N \
#       sharedfs=N tair_kvcm=N miss=N[ error=E]
#
# Only Scope.READ acquires reach ``_acquire_tiered_read``, so store-side
# creates stay silent.  Keys short-circuited by the connector's
# ``_cached_objs`` never reach the daemon and are not counted — same as
# production.  Contextvars are asyncio-task-local, so per-batch attribution
# stays exact even when the daemon interleaves concurrent acquires.
#
# Key lineage (sim-only, V6D_KEYSOURCE_LOG=1 by default):
#   * every read line gains a `` keys=k1,k2,...`` suffix (the probe keys);
#   * CREATE-scope acquires log ``[V6D CreateSource] request_id=<rid> n=N
#     keys=...`` — the intended writes of that request (keys the inner peer
#     merely touches because they already exist are included: for lineage
#     the requester is what matters);
#   * eviction batches log ``[V6D EvictSource] n=N keys=...``.
# Keys are daemon object keys (block-hash derived).  Joining them across
# requests yields the multi-turn ancestry: which request wrote / probed /
# evicted which boundary state.

_ENV_HITSOURCE_LOG = "V6D_HITSOURCE_LOG"
_ENV_KEYSOURCE_LOG = "V6D_KEYSOURCE_LOG"

_HITSOURCE_COUNTERS = ("local", "p2p", "sharedfs", "tair_kvcm")

# None -> outside a tracked read; dict -> current read's counter bag.
_hit_bag: contextvars.ContextVar = contextvars.ContextVar(
    "v6d_sim_hitsource_bag", default=None)


def _hitsource_enabled() -> bool:
    return os.environ.get(_ENV_HITSOURCE_LOG, "1") not in ("0", "false", "False")


def _keysource_enabled() -> bool:
    """Key-lineage logging: keys= suffix + CreateSource/EvictSource lines."""
    return os.environ.get(_ENV_KEYSOURCE_LOG, "1") not in ("0", "false", "False")


def _fmt_keys(object_keys) -> str:
    """Comma-joined, None-filtered key list (empty string when nothing)."""
    keys = [k for k in (object_keys or ()) if k]
    return ",".join(keys)


# ---------------------------------------------------------------------------
# Remote data-plane stubs (no real data movement in simulation)
# ---------------------------------------------------------------------------
#
# A P2P read in the real daemon resolves to two SRPC calls into the remote
# peer: ``transfer.get_metas_by_names`` (metadata probe, also used by
# ``VineyardPeer.get_remote_object_sizes`` for the eviction check) and
# ``transfer.load_data`` (the actual blob copy).
#
#   * ``load_data``/``async_load_data`` are ALWAYS no-ops.  They are only
#     reached by eager fetch strategies; the default lazy strategy never
#     calls them (data load is client-side, and the sim connector never
#     loads).

def _p2p_bandwidth_bps() -> float:
    raw = os.environ.get("HISIM_V6D_P2P_BANDWIDTH_GB", "").strip()
    if not raw:
        return 0.0
    try:
        gb_s = float(raw)
    except ValueError:
        raise ValueError("HISIM_V6D_P2P_BANDWIDTH_GB must be non-negative and finite") from None
    if not math.isfinite(gb_s) or gb_s < 0:
        raise ValueError("HISIM_V6D_P2P_BANDWIDTH_GB must be non-negative and finite")
    return gb_s * 1e9


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


def _p2p_base_s() -> float:
    raw = os.environ.get("HISIM_V6D_P2P_BASE_S", "").strip()
    if not raw:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        raise ValueError("HISIM_V6D_P2P_BASE_S must be non-negative and finite") from None
    if not math.isfinite(val) or val < 0:
        raise ValueError("HISIM_V6D_P2P_BASE_S must be non-negative and finite")
    return val


def _sim_load_data_noop(*args, **kwargs):
    """Skip the remote blob copy (eager strategies only; lazy never calls)."""
    bps = _p2p_bandwidth_bps()
    base_s = _p2p_base_s()
    if bps > 0 or base_s > 0:
        nbytes = _infer_transfer_bytes(args, kwargs)
        if nbytes > 0:
            delay = base_s + (nbytes / bps if bps > 0 else 0.0)
            if delay > 0:
                time.sleep(delay)
    logger.debug("[v6d-sim] skip remote blob data transfer (no data plane)")
    return None


def _sim_async_load_data_noop(*args, **kwargs):
    return []


class _PeerSocket:
    def __init__(self, native, address):
        self._native = native
        self._address = address

    def __getattr__(self, name):
        return getattr(self._native, name)

    def gethostbyname(self, hostname):
        if hostname == self._native.gethostname():
            return self._address
        return self._native.gethostbyname(hostname)


def _use_platform_peer_address(peer_method):
    address = os.environ.get("ALIYUN_ECI_ETH1_IP", "").strip()
    if not address:
        return
    ip = ipaddress.IPv4Address(address)
    if ip.is_loopback or ip.is_unspecified or ip.is_link_local or ip.is_multicast:
        raise ValueError("ALIYUN_ECI_ETH1_IP must be a usable IPv4 address")
    # Both peer classes resolve their advertised address through module globals.
    scope = peer_method.__globals__
    native = scope["socket"]
    if not isinstance(native, _PeerSocket):
        scope["socket"] = _PeerSocket(native, str(ip))
        logger.info("[v6d-network] advertise %s via ALIYUN_ECI_ETH1_IP", ip)


# ---------------------------------------------------------------------------
# C_VineyardPeerHook — argv patch
# ---------------------------------------------------------------------------

class C_VineyardPeerHook(BaseHook):
    """Inject ``reserve_memory=false`` via ``VineyardPeer.__init__`` argv and
    make SRPC init soft-fail.

    ``--vineyard-reserve-memory`` is a ``store_true`` CLI flag defaulting to
    True, so it cannot be disabled from the command line; swapping at argv
    level is the only way in.  With 2M alignment kept (the default), the
    SRPC server registers the bulkstore fine, so ``init_srpc_transfer`` runs
    for real — wrapped in try/except only so an unexpected failure degrades
    to "P2P meta probe fails" instead of a fatal boot error.
    """

    HOOK_CLASS_NAME = "VineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.vineyard.peer"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        _use_platform_peer_address(original_init)

        def patched_init(self, argc=0, argv=None, *args, **kwargs):

            import v6d.common.transfer as transfer

            _original_init_srpc = transfer.init_srpc_transfer

            def _init_srpc_soft_fail(*args, **kwargs):
                try:
                    return _original_init_srpc(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        "[v6d-sim] SRPC init failed (%s); daemon continues, "
                        "P2P meta probing will fail like a dead peer", e)
                    return None

            transfer.init_srpc_transfer = _init_srpc_soft_fail

            if argv is not None:
                argv = [
                    "--reserve_memory=false" if a == "--reserve_memory=true"
                    else a
                    for a in argv
                ]
            return original_init(self, argc, argv, *args, **kwargs)

        target.__init__ = patched_init
        logger.info(
            "[v6d-sim] C_VineyardPeerHook installed: reserve_memory=false, "
            "SRPC init soft-fail")


# ---------------------------------------------------------------------------
# C_TieredVineyardPeerHook — real read path + hit-source stats + data stubs
# ---------------------------------------------------------------------------

class C_TieredVineyardPeerHook(BaseHook):
    """Keep the real ``_acquire_tiered_read`` on the CPU read path.

    Two pieces, both daemon-process only (this class is defined only where
    the tiered peer module is imported, i.e. the daemon):

    1. Observability: wrap ``_acquire_tiered_read`` with a contextvar
       counter bag and log one ``[V6D HitSource]`` line per read batch — a
       direct port of the production ``v6d_hitsource_patch`` (hit_stats.sh).
    2. Data-plane stubs: no-op the two payload-copy entry points in
       ``v6d.common.transfer`` (eager strategies only — never called by the
       default lazy strategy).  The remote SRPC meta probe
       remains native. Everything else on the read path (LRU
       touch, lease pin, tracker probe, eviction check, local admit +
       re-ANNOUNCE, ``record_*_hit``) runs unmodified.
    """

    HOOK_CLASS_NAME = "TieredVineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.tiered_vineyard.peer"

    @classmethod
    def hook(cls, target):
        _use_platform_peer_address(target._acquire_tiered_read)
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
                    keys_suffix = (" keys=" + _fmt_keys(object_keys)
                                   if _keysource_enabled() else "")
                    logger.info(
                        "[V6D HitSource] request_id=%s queried=%d local=%d p2p=%d "
                        "sharedfs=%d tair_kvcm=%d miss=%d%s%s",
                        request_id, queried, bag["local"], bag["p2p"],
                        bag["sharedfs"], bag["tair_kvcm"],
                        max(queried - hits, 0),
                        f" error={error}" if error else "",
                        keys_suffix,
                    )

        target._acquire_tiered_read = wrapped_acquire_tiered_read

        # ---- key lineage: CREATE acquires + eviction batches -------------
        # ``acquire`` is the outer entrypoint (daemon /acquire route); CREATE
        # scope dispatches to the create path whose BATCH_CREATE keys are
        # exactly ``object_keys``.  ``request_id`` is keyword in every caller
        # seen, with a positional fallback for safety.
        original_acquire = target.acquire

        async def wrapped_acquire(self, object_keys, scope, *args, **kwargs):
            if not _keysource_enabled() or "CREATE" not in str(scope):
                return await original_acquire(self, object_keys, scope, *args, **kwargs)
            request_id = kwargs.get("request_id")
            if request_id is None:
                request_id = next(
                    (a for a in args[3:] if isinstance(a, str)), None)
            try:
                return await original_acquire(self, object_keys, scope, *args, **kwargs)
            finally:
                keys = _fmt_keys(object_keys)
                if keys:
                    logger.info(
                        "[V6D CreateSource] request_id=%s n=%d keys=%s",
                        request_id, len([k for k in (object_keys or ()) if k]), keys)

        target.acquire = wrapped_acquire

        # Eviction batches carry the objects being dropped — the death side
        # of the lineage.  Batched (~15 keys/call, ~30 calls/min/pod).
        # Extra args (e.g. the async evictor's ``is_emergency``) pass through.
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
                        len([k for k in (object_keys or ()) if k]), keys)

        target._evict_batch_from_vineyard = wrapped_evict_batch

        # The tiered peer module already imports v6d.common.transfer (through
        # v6d.server.peers.vineyard.peer), so this import is a sys.modules
        # lookup; the stubs apply process-wide, which is fine: the client
        # side never transfers data either (fetch is stubbed there).
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


# ---------------------------------------------------------------------------
# C_HitRateStatsHook — feed the per-read hit-source bag
# ---------------------------------------------------------------------------

class C_HitRateStatsHook(BaseHook):
    """Bump the contextvar bag from ``HitRateStats.record_*_hit``.

    Direct port of the production ``v6d_hitsource_patch`` stats hook; the
    daemon's own classification (local / remote / sharedfs / tair_kvcm) is
    reused unchanged, so the sim's split is defined by the same code as
    production.
    """

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_RUNTIME_HOOKS_INSTALLED = False


def install_v6d_runtime_hooks():
    """Apply all v6d-side simulation hooks before v6d.cli main (class hooks
    must precede the imports that define the target classes).

    Idempotent: the .pth bootstrap and the daemon entrypoint may both call it.
    """
    global _RUNTIME_HOOKS_INSTALLED
    if _RUNTIME_HOOKS_INSTALLED:
        return
    from sglang_simulator.simulation.vllm.v6d.tair_kvcm import (
        C_TairKVCMTransferRuntimeHook,
    )

    install_class_hooks([
        C_VineyardPeerHook,
        C_TairKVCMTransferRuntimeHook,
        C_TieredVineyardPeerHook,
        C_HitRateStatsHook,
    ])
    # The daemon can import transfer before the .pth bootstrap installs hooks.
    transfer = sys.modules.get(C_TairKVCMTransferRuntimeHook.HOOK_MODULE_NAME)
    runtime = getattr(transfer, C_TairKVCMTransferRuntimeHook.HOOK_CLASS_NAME, None)
    if runtime is not None:
        C_TairKVCMTransferRuntimeHook.hook(runtime)
    _RUNTIME_HOOKS_INSTALLED = True
