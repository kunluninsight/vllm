"""Keep vLLM's native v6d lookup/allocation while bypassing payload fetching.

Managers connect to the daemon; fetched placeholders complete the lease/seal
protocol without BlockReceiver. Native daemon hooks configure remote probes
separately. Scheduler finish hooks clean up CPU no-op stores.
"""

from __future__ import annotations
import sys

from hisim.hook import BaseHook
from hisim.utils import get_logger

logger = get_logger()


class C_V6dObjectManagerHook(BaseHook):
    """Connect through the native retry helper, then run native connect callbacks."""

    HOOK_CLASS_NAME = "V6dObjectManager"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        def override_async_connect(self) -> None:
            """Connect and notify the backend; log connection failures without raising."""
            try:
                _mod = sys.modules.get(target.__module__)
                _fn = getattr(_mod, "_connect_v6d_with_retry", None)
                if _fn is None:
                    raise RuntimeError("_connect_v6d_with_retry not found")
                self.client = _fn(self._v6d_url)
                logger.info(
                    "[V6D P2P] Manager group=%d connected to v6d daemon "
                    "at %s (real P2P path)",
                    self._group_id,
                    self._v6d_url,
                )
            except Exception:
                logger.exception(
                    "[V6D P2P] Manager group=%d cannot connect to v6d daemon at %s",
                    self._group_id,
                    getattr(self, "_v6d_url", "?"),
                )
                return
            # With VLLM_V6D_ASYNC_REGISTER=1 this pushes the KV cache spec; skipping it
            # leaves v6d at tair_kvcm.spec_received=false, never registered with KVCM.
            on_connected = getattr(self, "_on_connected", None)
            if on_connected is not None:
                try:
                    on_connected(self.client)
                except Exception:
                    logger.warning(
                        "[V6D P2P] Manager group=%d on_connected callback failed",
                        self._group_id,
                        exc_info=True,
                    )
            if self._v6d_backend is not None:
                self._v6d_backend.mark_manager_connected(self._group_id)

        target._async_connect = override_async_connect

        logger.info("[V6D Hijack] V6dObjectManager hook installed (real v6d P2P path)")


class C_V6dObjectFetchHelperHook(BaseHook):
    """Complete non-local placeholders by lease without fetching payload bytes."""

    HOOK_CLASS_NAME = "V6dObjectFetchHelper"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        def override_start_fetch(self, objs):
            objs = [o for o in (objs or ()) if o is not None]
            if not objs:
                return None
            by_lease: dict = {}
            for o in objs:
                lease_id = getattr(o, "_lease_id", None)
                if lease_id:
                    by_lease.setdefault(lease_id, []).append(o)
            for lease_id, group in by_lease.items():
                pending = [
                    o
                    for o in group
                    if getattr(o, "meta", {}).get("location", "local") != "local"
                ]
                if not pending:
                    continue
                try:
                    # Exclude probe-only objects from this lease's seal target.
                    pending[0].set_seal_target(len(pending))
                except Exception as e:
                    logger.warning(
                        "[V6D Hijack] sim fetch: set_seal_target failed "
                        "lease=%s: %r; %d placeholder(s) left to the "
                        "daemon's release/discard path",
                        lease_id,
                        e,
                        len(pending),
                    )
                    continue
                for o in pending:
                    try:
                        o.complete()
                    except Exception as e:
                        logger.warning(
                            "[V6D Hijack] sim fetch: complete failed key=%s: %r",
                            getattr(o, "key", "?"),
                            e,
                        )
            return None

        target.start_fetch = override_start_fetch
        logger.info(
            "[V6D Hijack] V6dObjectFetchHelper hook installed "
            "(seal-only fetch, no data movement)"
        )


class _CapturingLoadPlans(dict):
    """Snapshot each load plan on write.

    build_connector_meta on the core thread swaps ``_reqs_to_load`` for a new
    dict, so a plan published on the connector loop can vanish before the
    backend reads it ("Missing prepared load plan").
    """

    __slots__ = ("captured",)

    def __init__(self, data, captured):
        super().__init__(data)
        self.captured = captured

    def __setitem__(self, req_id, groups):
        self.captured[req_id] = {
            gid: (list(keys), list(block_ids))
            for gid, (keys, block_ids) in groups.items()
        }
        super().__setitem__(req_id, groups)

    def __reduce__(self):
        # Connector metadata reaches workers as a plain dict.
        return dict, (dict(self),)


def pop_captured_load_plan(scheduler, req_id):
    """Return and forget the plan snapshot captured for ``req_id``, if any."""
    captured = getattr(scheduler, "__dict__", {}).get("_sim_load_plans")
    return captured.pop(req_id, None) if captured is not None else None


def _get_reqs_to_load(self):
    try:
        return self.__dict__["_sim_reqs_to_load"]
    except KeyError:
        raise AttributeError("_reqs_to_load") from None


def _set_reqs_to_load(self, value):
    captured = self.__dict__.setdefault("_sim_load_plans", {})
    self.__dict__["_sim_reqs_to_load"] = _CapturingLoadPlans(value, captured)


class C_V6dObjectConnectorSchedulerHook(BaseHook):
    """Finish CPU no-op stores without waiting; leave matching/allocation native."""

    HOOK_CLASS_NAME = "V6dObjectConnectorScheduler"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        target._reqs_to_load = property(_get_reqs_to_load, _set_reqs_to_load)
        original_layout = getattr(target, "_compute_group_block_bytes", None)
        if original_layout is not None:

            def compute_group_block_bytes(self, spec, group_id):
                from hisim.simulation.hijack.vllm.kv_layout import layout_backend_view

                # Storage represents target shards, not the single CPU executor.
                return original_layout(layout_backend_view(self), spec, group_id)

            target._compute_group_block_bytes = compute_group_block_bytes
        original_request_finished = getattr(target, "request_finished", None)
        original_request_finished_all_groups = getattr(
            target, "request_finished_all_groups", None
        )

        def _complete_cpu_store_noop(self, req_id):
            if hasattr(self, "_pending_store_reqs"):
                self._pending_store_reqs.discard(req_id)
            if hasattr(self, "_finished_pending_store_reqs"):
                self._finished_pending_store_reqs.discard(req_id)
            if hasattr(self, "_storing_block_hashes"):
                self._storing_block_hashes.pop(req_id, None)
            try:
                from vllm.v1.hybrid_connector import (
                    mark_backend_save_done,
                    sched_get_req,
                )

                req = sched_get_req(req_id)
                if req is not None:
                    mark_backend_save_done(req)
            except Exception:
                pass
            if hasattr(self, "_release_protected_blocks"):
                self._release_protected_blocks(req_id)
            logger.debug(
                "[V6D RPC Bypass] Request %s: completed scheduler-side "
                "CPU no-op store without async wait",
                req_id,
            )

        def override_request_finished(self, request, block_ids):
            if original_request_finished is None:
                return False, None
            should_wait, params = original_request_finished(self, request, block_ids)
            if hasattr(self, "_saving"):
                self._saving.pop(request.request_id, None)
            if should_wait:
                _complete_cpu_store_noop(self, request.request_id)
                return False, params
            return should_wait, params

        def override_request_finished_all_groups(self, request, block_ids):
            if original_request_finished_all_groups is None:
                return False, None
            should_wait, params = original_request_finished_all_groups(
                self, request, block_ids
            )
            if should_wait:
                _complete_cpu_store_noop(self, request.request_id)
                return False, params
            return should_wait, params

        target.request_finished = override_request_finished
        target.request_finished_all_groups = override_request_finished_all_groups

        logger.info(
            "[V6D Hijack] V6dObjectConnectorScheduler hook installed "
            "(CPU no-op store completion)"
        )
