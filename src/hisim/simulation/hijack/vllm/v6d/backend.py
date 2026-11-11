"""Replace vLLM v6d/KVT payload transport with CPU completion modeling.

Native lookup, allocation, and acknowledgement paths remain active.
SimulatedKVController owns transfer deadlines and Mamba block pins;
CUDA events and blade_kvt operations use lightweight shims.
"""

import asyncio
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass

from hisim.env import Envs
from hisim.simulation.manager import ConfigManager, StateManager
from hisim.hook import BaseHook
from hisim.simulation.hijack.vllm.cpu_stubs import DummyEvent
from hisim.simulation.hijack.v6d.bandwidth import BandwidthModel
from hisim.simulation.hijack.vllm.v6d.manager import pop_captured_load_plan
from hisim.utils import get_logger

logger = get_logger()


def _ple_admission_model(vllm_config):
    """Configured admission model, disabled when the served model has no PLE.

    remote_v6d is an environment switch, so it can be set for any model.
    """
    from hisim.time_predictor.ple import PLEAdmissionModel, model_has_ple

    model = ConfigManager.get_ple_admission_model()
    model_config = getattr(vllm_config, "model_config", None)
    if (
        model.enabled
        and model_config is not None
        and not model_has_ple(model_config.hf_text_config)
    ):
        logger.info("[V6D Hijack] PLE admission disabled: model has no PLE layers")
        return PLEAdmissionModel(enabled=False)
    return model


class SimulatedKVController:
    """Own transfer deadlines and block pins, driven by native connector polling.

    Stage stores from metadata; progress schedules and acknowledges them using
    the wall clock or OFFLINE virtual time. No worker callbacks are needed.
    """

    @dataclass
    class Store:
        request: object
        blocks: list
        num_blocks: int
        last: bool
        ready_at: float = 0.0

    def __init__(self, hybrid):
        self.hybrid = hybrid
        self.backend = getattr(hybrid._backend, "_v6d", hybrid._backend)
        self.cache = self.backend._scheduler
        self.backend._sim_controller = self
        self._staged = []
        self._stores = []
        self._loads = {}
        self._ple = {}
        self._ple_model = _ple_admission_model(getattr(hybrid, "_cfg", None))
        self._store_tails = {}
        self._aborted_stores = {}
        self._preparing = []
        self._lock = threading.RLock()
        # Serialize native abort cleanup with transfer ownership of block pins.
        original_release = self.cache._release_protected_blocks

        def release(req_id):
            with self._lock:
                return original_release(req_id)

        self.cache._release_protected_blocks = release

    @staticmethod
    def now():
        return (
            StateManager.get_global_clock()
            if Envs.simulation_mode() == "OFFLINE"
            else time.perf_counter()
        )

    @staticmethod
    def _metadata_objects(metadata):
        seen, stack = set(), [metadata]
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            yield obj
            stack.extend(getattr(obj, attr, None) for attr in ("reqs", "inner", "v6d"))

    def capture_stores(self, metadata):
        """Detach this chunk's protected pins; defer transfer timing to progress."""
        stores = {}
        for obj in self._metadata_objects(metadata):
            stores.update(getattr(obj, "reqs_to_store", None) or {})
            for rid in getattr(obj, "aborted_save_ids", ()):
                state = self.hybrid._saving.get(rid)
                if state is not None:
                    self._aborted_stores[rid] = state._req
        # Store entries begin with (groups_data, is_last_save). Ignore trailing
        # transport fields such as create_batch_id, but validate the leading pair.
        _store_entries = tuple(stores.items())
        if _store_entries and not getattr(self, "_sim_store_arity_logged", False):
            _entry0 = _store_entries[0][1]
            if not isinstance(_entry0, (tuple, list)):
                raise TypeError(
                    "[V6D Hijack] reqs_to_store entry is "
                    f"{type(_entry0).__name__}, expected a sequence "
                    "(groups_data, is_last_save[, create_batch_id])"
                )
            _arity = len(_entry0)
            if _arity < 2:
                raise ValueError(
                    "[V6D Hijack] reqs_to_store entry arity "
                    f"{_arity} < 2: engine metadata contract changed, "
                    "cannot model store completion / save-done"
                )
            logger.info(
                "[V6D Hijack] reqs_to_store entry arity=%d "
                "(groups_data, is_last_save%s), indexing positionally",
                _arity,
                ", create_batch_id"
                if _arity == 3
                else (f", +{_arity - 2} trailing fields" if _arity > 3 else ""),
            )
            self._sim_store_arity_logged = True
        for rid, _entry in _store_entries:
            groups, last = _entry[0], _entry[1]
            state = self.hybrid._saving.get(rid)
            if state is None:
                raise RuntimeError(f"Store request {rid} is missing")
            # Group info begins with (keys, gpu_block_ids); newer engines append fields.
            wanted = Counter(
                bid
                for gid, info in groups.items()
                if gid in self.cache.mamba_group_ids
                for bid in info[1]
            )
            with self._lock:
                taken = self._detach_protected_blocks(rid, wanted)
            self._staged.append(
                self.Store(
                    state._req,
                    taken,
                    sum(len(info[0]) for info in groups.values()),
                    last,
                )
            )

    def _detach_protected_blocks(self, rid, wanted):
        """Take wanted pins from per-request state or the legacy scheduler dict."""
        legacy = getattr(self.cache, "_swap_protected_blocks", None)
        if legacy is None:
            state = self.cache._get_state(rid)
            pinned = state.protected_blocks if state is not None else ()
        else:
            pinned = legacy.get(rid, ())
        taken, remaining = [], []
        for block in pinned:
            if wanted[block.block_id]:
                wanted[block.block_id] -= 1
                taken.append(block)
            else:
                remaining.append(block)
        if legacy is None:
            if state is not None:
                state.protected_blocks = remaining
        elif remaining:
            legacy[rid] = remaining
        else:
            legacy.pop(rid, None)
        return taken

    def prepare(self, coroutine):
        future = Future()
        self._preparing.append(future)

        async def tracked():
            try:
                result = await coroutine
            except BaseException as exc:
                future.set_exception(exc)
                raise
            else:
                future.set_result(result)
                return result

        return tracked()

    def settle_preparations(self):
        """Wait for control-plane RPCs (OFFLINE deterministic barrier)."""
        preparing, self._preparing = self._preparing, []
        for future in preparing:
            future.result(timeout=30.0)

    def reap_preparations(self):
        """Publish completed RPCs without blocking the EngineCore thread."""
        pending = []
        for future in self._preparing:
            if future.done():
                # Surface connector-loop failures on the engine thread.
                future.result()
            else:
                pending.append(future)
        self._preparing = pending

    def queue_load(self, request, num_tokens, groups):
        nblocks = sum(len(keys) for keys, _ids in groups.values())
        bw = BandwidthModel.get()
        deadline = self.now() + bw.latency_for(nblocks, True) + bw.seg1_latency(nblocks)
        with self._lock:
            self._loads.setdefault(request.request_id, (deadline, num_tokens))

    def queue_ple(self, request, run) -> bool:
        """Defer ``run`` (native mark_loaded) by the remote PLE admission time.

        Returns False when no delay applies, so the caller runs it inline.
        """
        latency = self._ple_model.latency(request.num_prompt_tokens)
        if latency <= 0:
            return False
        with self._lock:
            if request.request_id in self._ple:
                raise RuntimeError(f"PLE admission for {request.request_id} is queued")
            self._ple[request.request_id] = (self.now() + latency, run)
        return True

    def _run_control(self, coroutine):
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is self.hybrid.loop:
            coroutine.close()
            raise RuntimeError(
                "Cannot synchronously wait on the current connector loop"
            )
        return asyncio.run_coroutine_threadsafe(coroutine, self.hybrid.loop).result(
            timeout=30.0
        )

    def _load_done(self, rid, num_tokens):
        from vllm.v1.hybrid_connector import IoRet

        async def complete():
            for rank in range(self.hybrid._tp_size()):
                await self.hybrid._do_load_done(rank, IoRet(reqid=rid, n=num_tokens))

        self._run_control(complete())

    def _store_done(self, req, failed=False):
        from vllm.v1.hybrid_connector import HB_SAVE_SOURCES, IoRet, get_param

        sources = tuple(get_param(req, HB_SAVE_SOURCES, ()) or ())
        if not sources:
            raise RuntimeError(f"Missing save sources for {req.request_id}")

        async def complete():
            for source in sources:
                for rank in range(self.hybrid._tp_size()):
                    await self.hybrid._do_save_done(
                        rank,
                        IoRet(
                            reqid=req.request_id, source=source, n=0 if failed else None
                        ),
                    )

        self._run_control(complete())

    def progress(self):
        """Settle work through this clock; future readiness stays private."""
        now = self.now()
        bw = BandwidthModel.get()
        staged, self._staged = self._staged, []
        for store in staged:
            rid = store.request.request_id
            store.ready_at = max(now, self._store_tails.get(rid, now))
            store.ready_at += bw.store_completion_latency(store.num_blocks)
            self._store_tails[rid] = store.ready_at
            self._stores.append(store)
        due = [store for store in self._stores if store.ready_at <= now]
        self._stores = [store for store in self._stores if store.ready_at > now]
        for store in due:
            if store.blocks:
                with self._lock:
                    self.cache._block_pool.free_blocks(store.blocks)
            rid = store.request.request_id
            if self._store_tails.get(rid) == store.ready_at:
                self._store_tails.pop(rid, None)
            if store.last and rid not in self._aborted_stores:
                self._store_done(store.request)
        aborted = [rid for rid in self._aborted_stores if rid not in self._store_tails]
        for rid in aborted:
            self._store_done(self._aborted_stores.pop(rid), failed=True)
        with self._lock:
            loads = [
                (rid, n)
                for rid, (deadline, n) in self._loads.items()
                if deadline <= now
            ]
            for rid, _n in loads:
                del self._loads[rid]
        for rid, n in loads:
            self._load_done(rid, n)
        # Loads settled above queue PLE with a future deadline, never due here.
        with self._lock:
            ples = [
                (rid, run)
                for rid, (deadline, run) in self._ple.items()
                if deadline <= now
            ]
            for rid, _run in ples:
                del self._ple[rid]
        for _rid, run in ples:
            self._run_control(run())
        return len(due) + len(loads) + len(aborted) + len(ples)

    def next_wakeup(self):
        if self._staged or any(
            rid not in self._store_tails for rid in self._aborted_stores
        ):
            return self.now()
        with self._lock:
            deadlines = [deadline for deadline, _n in self._loads.values()]
            deadlines.extend(deadline for deadline, _run in self._ple.values())
        deadlines.extend(store.ready_at for store in self._stores)
        return min(deadlines, default=None)

    def has_pending(self):
        return bool(self._preparing) or self.next_wakeup() is not None

    def reset(self):
        """Refuse active requests; release pending pins and fail unfinished saves.

        Called by connector cache reset, not profile/statistics reset.
        Real control-plane preparations are settled without advancing virtual time.
        """
        if (
            self.hybrid.pending_requests
            or self.hybrid._loaded
            or any(
                not state._req.is_finished() for state in self.hybrid._saving.values()
            )
        ):
            logger.warning("[V6D Hijack] controller reset refused: requests active")
            return False
        self.settle_preparations()
        with self._lock:
            dropped_loads = len(self._loads) + len(self._ple)
            self._loads.clear()
            self._ple.clear()
            for store in self._staged + self._stores:
                if store.blocks:
                    self.cache._block_pool.free_blocks(store.blocks)
            self._staged.clear()
            self._stores.clear()
            self._store_tails.clear()
            self._aborted_stores.clear()
        if dropped_loads:
            logger.warning(
                "[V6D Hijack] controller reset dropped %d pending load(s)",
                dropped_loads,
            )
        saved = set(self.hybrid._saved)
        for rid, state in list(self.hybrid._saving.items()):
            if rid not in saved:
                self._store_done(state._req, failed=True)
        self.hybrid._step_saved()
        return not self.has_pending()


class C_HybridControlPlaneHook(BaseHook):
    """Drive modeled transfers alongside native admission and metadata preparation."""

    HOOK_CLASS_NAME = "HybridScheduler"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_prepare = target._on_add_req
        original_waiting = target._step_waiting
        original_step = target.step
        original_mark_loaded = getattr(target, "mark_loaded", None)

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            backend = getattr(self._backend, "_v6d", self._backend)
            self._sim_controller = (
                SimulatedKVController(self)
                if getattr(backend, "_sim_sync_prepare", False)
                else None
            )

        def prepare(self, req, blocks):
            coroutine = original_prepare(self, req, blocks)
            controller = self._sim_controller
            return (
                controller.prepare(coroutine) if controller is not None else coroutine
            )

        def step_waiting(self):
            result = original_waiting(self)
            controller = self._sim_controller
            if controller is not None:
                if Envs.simulation_mode() == "OFFLINE":
                    # Settle real RPCs before advancing OFFLINE transfer readiness.
                    controller.settle_preparations()
                else:
                    # BLOCKING leaves RPCs asynchronous while the engine polls.
                    controller.reap_preparations()
                controller.progress()
            return result

        def step(self):
            if self._sim_controller is not None:
                self._sim_controller.reap_preparations()
                self._sim_controller.progress()
            return original_step(self)

        target.__init__ = init
        target._on_add_req = prepare
        target._step_waiting = step_waiting
        target.step = step

        async def mark_loaded(self, req, ioret=None, *args, **kwargs):
            # Native Prefill awaits the remote PLE fetch here, after KV load.
            controller = self._sim_controller
            if (
                controller is None
                or getattr(self, "_ple", None) is None
                or (ioret is not None and ioret.ex is not None)
                or getattr(req, "ple_ready", None) is not None
                or _req_aborted(req)
            ):
                return await original_mark_loaded(self, req, ioret, *args, **kwargs)
            # Native stamps KV completion before awaiting PLE; keep those times.
            kv_finished_at = time.monotonic()
            if getattr(
                req, "worker_status_remote_kv_wait_started_time_ms", 0
            ) > 0 and not getattr(
                req, "worker_status_remote_kv_wait_finished_time_ms", 0
            ):
                req.worker_status_remote_kv_wait_finished_time_ms = int(
                    time.time() * 1000
                )

            async def run():
                await original_mark_loaded(self, req, ioret, *args, **kwargs)
                req.ple_kv_finished_at = kv_finished_at

            if not controller.queue_ple(req, run):
                await run()

        if original_mark_loaded is not None:
            target.mark_loaded = mark_loaded


def _req_aborted(req) -> bool:
    from vllm.v1.hybrid_connector import req_aborted

    return req_aborted(req)


class C_HybridBackendHook(BaseHook):
    """Bypass GPU-oriented HybridBackend group layout validation on CPU."""

    HOOK_CLASS_NAME = "HybridBackend"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target):
        def override_validate_group_ordering(self):
            logger.info(
                "[V6D Hijack] HybridBackend._validate_group_ordering: "
                "skipped CPU native control-plane validation"
            )
            return None

        target._validate_group_ordering = override_validate_group_ordering
        logger.info("[V6D Hijack] HybridBackend hook installed")


class C_HybridConnectorHook(BaseHook):
    """Own modeled transfers while preserving native completion semantics."""

    HOOK_CLASS_NAME = "HybridConnector"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target):
        original_has_requests = getattr(target, "has_requests", None)
        original_build_meta = getattr(target, "build_connector_meta", None)

        def build_connector_meta(self, scheduler_output):
            metadata = original_build_meta(self, scheduler_output)
            controller = self._sched._sim_controller
            if controller is not None:
                controller.capture_stores(metadata)
            return metadata

        if original_build_meta is not None:
            target.build_connector_meta = build_connector_meta

        def has_requests(self):
            controller = self._sched._sim_controller
            return original_has_requests(self) or (
                controller is not None and controller.has_pending()
            )

        if original_has_requests is not None:
            target.has_requests = has_requests

        def override_wait_for_save(self):
            return None

        def override_get_finished(self, finished_req_ids):
            # Hybrid completion uses native acknowledgements, not base polling sets.
            return set(), set()

        def override_clear_connector_metadata(self):
            if getattr(self, "_worker", None) is not None:
                self._worker.clear_connector_metadata()
            return None

        def override_start_load_kv(self, forward_context=None, **kwargs):
            # Keep worker dispatch; the controller owns load acknowledgement.
            assert self._worker is not None
            self._worker.start_load_kv()

        target.wait_for_save = override_wait_for_save
        target.get_finished = override_get_finished
        target.clear_connector_metadata = override_clear_connector_metadata
        target.start_load_kv = override_start_load_kv

        def override_reset_cache(self):
            # Settle controller-owned pins/saves before resetting backend managers.
            controller = getattr(self._sched, "_sim_controller", None)
            if controller is not None and not controller.reset():
                logger.warning(
                    "[V6D Hijack] HybridConnector.reset_cache: controller "
                    "still has active requests; caches NOT reset"
                )
                return False
            backend = getattr(getattr(self, "_sched", None), "_backend", None)
            for candidate in (backend, getattr(backend, "_v6d", None)):
                scheduler = getattr(candidate, "_scheduler", None)
                if scheduler is not None and hasattr(scheduler, "reset_cache"):
                    result = scheduler.reset_cache()
                    logger.info(
                        "[V6D Hijack] HybridConnector.reset_cache: forwarded "
                        "to %s -> %s",
                        type(scheduler).__name__,
                        result,
                    )
                    return result
            logger.warning(
                "[V6D Hijack] HybridConnector.reset_cache: no backend "
                "scheduler with reset_cache found (backend=%s); connector "
                "caches NOT reset",
                type(backend).__name__ if backend else None,
            )
            return False

        target.reset_cache = override_reset_cache
        logger.info("[V6D Hijack] HybridConnector hook installed")


class C_V6dObjectBackendHook(BaseHook):
    """Use dummy CUDA events; let the controller time native load/save completion."""

    HOOK_CLASS_NAME = "V6dObjectBackend"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector.v6d_object_backend"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def override_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if hasattr(self, "_save_event_pool"):
                self._save_event_pool = [DummyEvent() for _ in range(8)]
            if hasattr(self, "_load_event_pool"):
                self._load_event_pool = [DummyEvent() for _ in range(8)]
            if getattr(self, "_scheduler", None) is not None and hasattr(
                self, "_v6d_ready"
            ):
                self._v6d_ready = True
                logger.info(
                    "[V6D Hijack] V6dObjectBackend scheduler marked ready "
                    "for CPU native control-plane"
                )
            logger.info(
                "[V6D Hijack] V6dObjectBackend.__init__: "
                "replaced CUDA Event pools with DummyEvent"
            )

        target.__init__ = override_init

        def override_new_event(self):
            """Return DummyEvent instead of torch.cuda.Event."""
            return DummyEvent()

        if hasattr(target, "_new_event"):
            target._new_event = override_new_event

        def override_record_event(self, event=None):
            """Record the supplied event, or create and return a dummy event."""
            if event is None:
                event = DummyEvent()
            event.record()
            return event

        if hasattr(target, "_record_event"):
            target._record_event = override_record_event

        original_prepare = getattr(target, "async_update_state_after_alloc", None)

        async def prepare_load(self, request, blocks, num_external_tokens):
            # Finish lookup/allocation, then queue a deadline without awaiting DMA.
            scheduler = self._scheduler
            rid = request.request_id
            # Drop a snapshot left by an earlier (e.g. preempted) round.
            pop_captured_load_plan(scheduler, rid)
            result = await original_prepare(self, request, blocks, num_external_tokens)
            captured = pop_captured_load_plan(scheduler, rid)
            if result is not None or num_external_tokens <= 0:
                return result
            if hasattr(scheduler, "_reqs_to_load"):
                groups = captured or scheduler._reqs_to_load.get(rid)
            elif hasattr(scheduler, "_get_state"):
                # Newer vLLM keeps the prepared load plan in per-request state.
                state = scheduler._get_state(rid)
                groups = state.pending_load_meta if state is not None else None
            else:
                raise RuntimeError(
                    "[V6D Hijack] Unsupported scheduler load-plan contract"
                )
            if not groups:
                raise RuntimeError(
                    f"[V6D Hijack] Missing prepared load plan for {rid} "
                    f"with {num_external_tokens} external tokens"
                )
            # Preserve native ownership while snapshotting every group for timing.
            groups = {
                gid: (list(keys), list(block_ids))
                for gid, (keys, block_ids) in groups.items()
            }
            self._sim_controller.queue_load(request, num_external_tokens, groups)
            return result

        if original_prepare is not None:
            target.async_update_state_after_alloc = prepare_load
            target._sim_sync_prepare = True

        async def override_async_load_kv(self, m):
            # Do not yield: the controller sends LOAD_DONE at the modeled deadline.
            return
            yield  # Preserve the native async-generator interface.

        target.async_load_kv = override_async_load_kv

        def clear_backend_metadata(self):
            # Store completion belongs to the controller, not worker metadata.
            self._bound_meta = None

        def bypass_bind(self, metadata):
            # Scheduler metadata capture handles aborts; bypass must not complete early.
            return None

        target.clear_backend_metadata = clear_backend_metadata
        target.bypass_bind = bypass_bind

        _bwm = BandwidthModel.get()
        logger.info(
            "[V6D Hijack] V6dObjectBackend hook installed "
            "(transfer-latency model enabled=%s)",
            _bwm.enabled,
        )


class _DummyBladeKVTModule:
    """Minimal blade_kvt shim used before the real CUDA extension is loaded."""

    @staticmethod
    def set_envs(*args, **kwargs):
        logger.info("[KVT Hijack] skip blade_kvt.set_envs")
        return None

    @staticmethod
    def is_nv_gpu(*args, **kwargs):
        return True


class _DummyKVTClient:
    """No-op blade_kvt client for CPU native-control-plane simulation."""

    def record_event(self, *args, **kwargs):
        return None

    def start_req_send(self, *args, **kwargs):
        return None

    def start_send_substep(self, *args, **kwargs):
        return None

    def submit_delta_send(self, *args, **kwargs):
        return None

    def submit_req_send2(self, *args, **kwargs):
        return None

    def start_send_step(self, *args, **kwargs):
        return None

    def flush_send_step(self, *args, **kwargs):
        return None

    async def send_error_done_req(self, *args, **kwargs):
        return None


class C_KVTPBackendHook(BaseHook):
    """Hook KVT PBackend worker-side data transport for CPU simulation."""

    HOOK_CLASS_NAME = "PBackend"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector.kvtbackend"

    @classmethod
    def hook(cls, target):
        module = sys.modules.get(target.__module__)
        if module is not None:

            def _noop_generate_nic_affinity(*args, **kwargs):
                logger.info("[KVT Hijack] skip generate_nic_affinity")
                return None

            module.generate_nic_affinity = _noop_generate_nic_affinity
            module.blade_kvt = _DummyBladeKVTModule

        original_init = target.__init__

        def override_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if getattr(self, "_bladkv_cli", None) is None:
                self._bladkv_cli = _DummyKVTClient()
            logger.info("[KVT Hijack] PBackend initialized with dummy transport")

        def override_register_kv_caches(self, kv_caches):
            self._bladkv_cli = _DummyKVTClient()
            logger.info(
                "[KVT Hijack] PBackend.register_kv_caches skipped "
                "blade_kvt client creation; layers=%d",
                len(kv_caches),
            )
            return None

        def override_async_save_kv_layer(self, *args, **kwargs):
            return None

        def override_bind_backend_metadata(self, *args, **kwargs):
            return None

        def override_bypass_bind(self, *args, **kwargs):
            return None

        def override_clear_backend_metadata(self, *args, **kwargs):
            return None

        def override_bypass_clear(self, *args, **kwargs):
            return None

        target.__init__ = override_init
        target.register_kv_caches = override_register_kv_caches
        target.async_save_kv_layer = override_async_save_kv_layer
        target.bind_backend_metadata = override_bind_backend_metadata
        target.bypass_bind = override_bypass_bind
        target.clear_backend_metadata = override_clear_backend_metadata
        target.bypass_clear = override_bypass_clear

        logger.info("[KVT Hijack] PBackend hook installed")
