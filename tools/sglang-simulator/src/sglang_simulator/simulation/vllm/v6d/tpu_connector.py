"""TPU connector CPU data-plane adapters.

Scheduler-side lookup and connector metadata remain native. Worker-side data
movement is replaced by explicit load/store completion sets so replay can run
without CUDA/TPU memory, SRPC, RDMA, or a real decode peer.

多组 Raiden producer 是上述原生 scheduler 策略的例外：单 CPU、无真实
D 端时使用本地完成契约；V6D sibling 的查询、保存和延迟释放仍由原生管理。
"""

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from functools import wraps
import math
import os
import time
import weakref
from typing import Callable

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()


def _metadata_with(args, kwargs, field):
    for value in (*args, *kwargs.values()):
        if hasattr(value, field):
            return value
    return None


@dataclass
class _TPUTransferGate:
    # Set only by the modeled device-ready RPC response, never a log timestamp.
    ready_at: float | None = None
    wait_ready: Callable[[], object] | None = None


_transfer_gate = ContextVar("tpu_transfer_gate", default=None)
_idle_wait_cap_s = ContextVar("tpu_idle_wait_cap_s", default=None)
_store_queues = weakref.WeakSet()
_timed_queues = weakref.WeakSet()


@contextmanager
def tpu_idle_wait_cap(max_wait_s: float | None):
    """Cap per-step idle polling sleep so single-process serving stays responsive."""
    token = _idle_wait_cap_s.set(max_wait_s)
    try:
        yield
    finally:
        _idle_wait_cap_s.reset(token)


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonneg_env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a non-negative finite number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def _tpu_load_bytes_per_second() -> float:
    gb_s = _nonneg_env_float("HISIM_TPU_V6D_LOAD_BANDWIDTH_GB", 0.0)
    return gb_s * 1e9 if gb_s > 0 else 0.0


def _tpu_raiden_handoff_timing() -> tuple[float, float]:
    base_s = _nonneg_env_float("HISIM_TPU_RAIDEN_HANDOFF_BASE_S", 0.0)
    bw_gb = _nonneg_env_float("HISIM_TPU_RAIDEN_HANDOFF_BANDWIDTH_GB", 0.0)
    return base_s, (bw_gb * 1e9 if bw_gb > 0 else 0.0)


@dataclass
class _TimedReqJob:
    req_id: str
    deadline: float


class _TimedReqQueue:
    # FIFO timed completion queue for H2D V6D loads and Raiden handoff holds.

    def __init__(self, lanes: int = 1):
        self.jobs = deque()
        self._lanes = [0.0] * max(1, lanes)
        _timed_queues.add(self)

    def enqueue(self, req_id: str, duration: float, now: float) -> float:
        idx = min(range(len(self._lanes)), key=self._lanes.__getitem__)
        deadline = max(now, self._lanes[idx]) + duration
        self._lanes[idx] = deadline
        self.jobs.append(_TimedReqJob(req_id, deadline))
        return deadline

    def harvest(self, now: float) -> set[str]:
        ready = set()
        if not self.jobs:
            return ready
        remaining = deque()
        for job in self.jobs:
            if job.deadline <= now:
                ready.add(job.req_id)
            else:
                remaining.append(job)
        self.jobs = remaining
        return ready

    def earliest_deadline(self) -> float | None:
        if not self.jobs:
            return None
        return min(job.deadline for job in self.jobs)


@contextmanager
def tpu_transfer_step():
    """Bind stores submitted by this execute_model to its PP device-ready gate."""
    gate = _TPUTransferGate()
    token = _transfer_gate.set(gate)
    try:
        yield gate
    finally:
        _transfer_gate.reset(token)


@dataclass
class _TPUStoreJob:
    job_id: int
    req_id: str
    gate: _TPUTransferGate
    duration: float
    deadline: float | None = None


class _TPUStoreQueue:
    def __init__(self):
        self.jobs = deque()
        self.seen = {}
        self.highest_job_id = -1
        self.lane_ready_at = 0.0
        self._num_lanes = _positive_env_int("HISIM_TPU_V6D_STORE_LANES", 1)
        self._lanes = [0.0] * self._num_lanes
        _store_queues.add(self)

    def resolve_deadlines(self):
        for job in self.jobs:
            if job.deadline is not None:
                continue
            if job.gate.ready_at is None:
                if self._num_lanes == 1:
                    break
                continue
            if not math.isfinite(job.gate.ready_at):
                raise ValueError("TPU transfer ready_at must be finite")
            if self._num_lanes == 1:
                job.deadline = max(job.gate.ready_at, self.lane_ready_at) + job.duration
                self.lane_ready_at = job.deadline
                self._lanes[0] = job.deadline
            else:
                idx = min(range(self._num_lanes), key=self._lanes.__getitem__)
                job.deadline = max(job.gate.ready_at, self._lanes[idx]) + job.duration
                self._lanes[idx] = job.deadline
                self.lane_ready_at = max(self._lanes)

    def earliest_deadline(self) -> float | None:
        deadlines = [j.deadline for j in self.jobs if j.deadline is not None]
        return min(deadlines) if deadlines else None


def wait_for_tpu_store_progress():
    """Wait for the earliest store/load/handoff event after PP gates resolve.

    Queues are worker-owned; weak registration never keeps an old engine alive.
    Nonempty execute_model and unresolved PP futures must never wait here.
    """
    queues = list(_store_queues)
    if any(job.gate.ready_at is None for q in queues for job in q.jobs):
        return
    deadlines = []
    for queue in queues:
        queue.resolve_deadlines()
        earliest = queue.earliest_deadline()
        if earliest is not None:
            deadlines.append(earliest)
    for tq in list(_timed_queues):
        earliest = tq.earliest_deadline()
        if earliest is not None:
            deadlines.append(earliest)
    if deadlines:
        delay = min(deadlines) - time.monotonic()
        cap = _idle_wait_cap_s.get()
        if cap is not None and delay > cap:
            delay = cap
        if delay > 0:
            time.sleep(delay)


def _tpu_node_object_layout(vllm_config, kv_cache_config):
    """Shared scheduler/worker geometry; no extra PP or PCP byte multiplier."""
    from sglang_simulator.simulation.manager import ConfigManager
    from sglang_simulator.simulation.vllm.tpu_kv_geometry import (
        _positive_int,
        is_tpu_target,
    )

    raw = ConfigManager._get_raw_config()
    scheduler = raw.get("scheduler", {})
    if (
        not is_tpu_target(raw)
        or scheduler.get("pp_size", 1) == 1
        or vllm_config is None
        or kv_cache_config is None
    ):
        return None
    from vllm.v1.core.kv_cache_utils import (
        _pool_bytes_per_block,
        _project_kv_cache_groups_to_worker,
    )
    from sglang_simulator.simulation.vllm.tpu_kv_capacity import (
        _parse_partition,
        _project_layers,
    )

    if vllm_config.parallel_config.local_world_size != 1:
        raise ValueError("TPU V6D folding requires one physical CPU worker")
    groups = kv_cache_config.kv_cache_groups
    specs = {
        name: group.kv_cache_spec for group in groups for name in group.layer_names
    }
    stages = _positive_int(scheduler["pp_size"], "scheduler.pp_size")
    partition = _parse_partition(
        scheduler, vllm_config.model_config.hf_text_config.num_hidden_layers, stages
    )
    pages = [
        _pool_bytes_per_block(_project_kv_cache_groups_to_worker(groups, stage))
        for stage in _project_layers(specs, partition)
    ]
    ranks = _positive_int(raw["platform"]["num_device_per_node"], "num_device_per_node")
    if ranks != scheduler.get("tp_size", 1) * scheduler.get("pcp_size", 1):
        raise ValueError("TPU V6D requires one complete PP stage per target node")
    page = _positive_int(max(pages), "TPU stage page bytes")
    logger.info(
        "[TPU V6D capacity] target_nodes=%d target_local_ranks=%d stage_page_bytes=%s "
        "object_bytes=%d; physical_daemons=1, node readiness remains folded",
        stages, ranks, pages, ranks * page,
    )
    return ranks * page, [ranks, 1, page]


def _tpu_store_bytes_per_second():
    from sglang_simulator.simulation.manager import ConfigManager

    value = ConfigManager._get_raw_config().get("platform", {}).get(
        "memory_write_bandwidth_gb"
    )
    if isinstance(value, bool):
        raise ValueError("TPU memory_write_bandwidth_gb must be positive and finite")
    try:
        bandwidth = float(value) * 1e9  # PlatformConfig uses decimal GB.
    except (TypeError, ValueError):
        raise ValueError("TPU memory_write_bandwidth_gb must be positive and finite") from None
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("TPU memory_write_bandwidth_gb must be positive and finite")
    return bandwidth


@dataclass(frozen=True)
class _LocalRaidenHandoff:
    group_block_ids: tuple[tuple[int, ...], ...]
    num_tokens: int


class C_TPUV6dObjectLayoutHook(BaseHook):
    """Charge the folded local cache for one complete target node's page.

    This fixes capacity accounting. The single physical daemon still cannot
    express independent node failures or differences in publication readiness.
    """

    HOOK_MODULE_NAME = "kvs_connector.v6d_object_connector"
    HOOK_CLASS_NAME = "V6dObjectConnectorScheduler"

    @classmethod
    def hook(cls, target):
        original = target._compute_group_block_bytes

        def compute(self, spec, group_id):
            result = original(self, spec, group_id)
            if not hasattr(self, "_sim_tpu_v6d_layout"):
                layout = _tpu_node_object_layout(
                    getattr(self, "vllm_config", None),
                    getattr(self, "kv_cache_config", None),
                )
                if layout is None:
                    return result
                self._sim_tpu_v6d_layout = (*layout, result[2])
            size, shape, dtype = self._sim_tpu_v6d_layout
            return size, list(shape), dtype

        target._compute_group_block_bytes = compute


class C_MultiKVConnectorPromMetricsHook(BaseHook):
    """Preserve native metric values when CPU execution skips IPC serialization."""

    HOOK_MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    HOOK_CLASS_NAME = "MultiKVConnectorPromMetrics"

    @classmethod
    def hook(cls, target):
        if target.__dict__.get("_sim_nested_stats_hook", False):
            return

        original = target.observe

        @wraps(original)
        def observe(self, transfer_stats_data, engine_idx=0):
            # Match native IPC serialization, including nested connector stats.
            serialized = {
                name: asdict(stats) if is_dataclass(stats) else stats
                for name, stats in transfer_stats_data.items()
            }
            return original(self, serialized, engine_idx)

        target.observe = observe
        target._sim_nested_stats_hook = True


class C_TPUV6dConnectorAdmissionHook(BaseHook):
    """Observe the native accepted external prefix before async metadata expires.

    New TPU connectors can finish the load on an idle engine step. The later
    first forward then has no reqs_to_load metadata, so metadata-only inference
    would mislabel the V6D prefix as an L1 hit. No cache decisions are changed.
    """

    HOOK_MODULE_NAME = "kvs_connector.v6d_object_connector"
    HOOK_CLASS_NAME = "TPUV6dObjectConnector"

    @classmethod
    def hook(cls, target):
        original = target.update_state_after_alloc

        def allocated(self, request, blocks, num_external_tokens):
            first_admission = (request.num_computed_tokens == 0
                               and getattr(request, "num_preemptions", 0) == 0)
            result = original(self, request, blocks, num_external_tokens)
            if first_admission:
                from sglang_simulator.simulation.req_stats_manager import (
                    request_stats_manager,
                )
                if request.request_id in request_stats_manager.stats:
                    request_stats_manager.external_prefix_tokens.setdefault(
                        request.request_id, max(num_external_tokens, 0)
                    )
            return result

        original_pending = getattr(target, "has_pending_push_work", None)

        def has_pending_push_work(self):
            if original_pending is not None and original_pending(self):
                return True
            sched = getattr(self, "scheduler", None)
            if sched is not None and (
                getattr(sched, "_store_jobs", None)
                or getattr(sched, "_store_waits", None)
            ):
                return True
            worker = getattr(self, "worker", None)
            if worker is not None:
                worker_pending = getattr(worker, "has_pending_push_work", None)
                if worker_pending is not None and worker_pending():
                    return True
            return False

        target.update_state_after_alloc = allocated
        target.has_pending_push_work = has_pending_push_work


class C_TPURaidenConnectorHook(BaseHook):
    """单 CPU 多组 producer 的本地完成契约，不模拟真实 P→D 交接。"""

    HOOK_MODULE_NAME = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    HOOK_CLASS_NAME = "TPURaidenConnector"

    @classmethod
    def hook(cls, target):
        if target.__dict__.get("_sim_local_raiden_hook", False):
            return
        original_init = target.__init__
        original_admit = target.on_new_request
        original_match = target.get_num_new_matched_tokens
        original_alloc = target.update_state_after_alloc
        original_meta = target.build_connector_meta
        original_finish = target.request_finished
        original_finish_groups = target.request_finished_all_groups
        original_update = target.update_connector_output
        original_pending = getattr(target, "has_pending_push_work", None)

        def init(self, vllm_config, role, kv_cache_config):
            groups = kv_cache_config.kv_cache_groups
            transfer = vllm_config.kv_transfer_config
            local = transfer.kv_role == "kv_producer" and len(groups) > 1
            if local:
                parallel = vllm_config.parallel_config
                dimensions = (
                    "tensor_parallel_size", "pipeline_parallel_size",
                    "prefill_context_parallel_size", "decode_context_parallel_size",
                    "data_parallel_size", "nnodes",
                )
                if any(getattr(parallel, name, 1) != 1 for name in dimensions):
                    raise ValueError("Local Raiden simulation requires single-worker topology")
            original_init(self, vllm_config, role, kv_cache_config)
            self._sim_local_raiden = local
            self._sim_raiden_num_groups = len(groups)
            self._sim_raiden_pending_sends = {}
            self._sim_raiden_inflight = {}
            if local:
                from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
                    MambaSpec,
                    _stage3_group_window_tokens,
                )
                self._sim_raiden_truncates_prompt = any(
                    isinstance(group.kv_cache_spec, MambaSpec)
                    or _stage3_group_window_tokens(group.kv_cache_spec)
                    for group in groups
                )
            if local and self.connector_scheduler is not None:
                logger.info(
                    "[Raiden CPU] local-only producer enabled: kv_groups=%d; "
                    "local handoff pin/ACK only, no remote transfer timing; V6D lifecycle unchanged",
                    len(groups),
                )

        def admit(self, request):
            if not self._sim_local_raiden:
                return original_admit(self, request)
            params = getattr(request, "kv_transfer_params", None)
            if params is not None and not isinstance(params, dict):
                raise ValueError("Local Raiden simulation requires dict kv_transfer_params")
            remote_fields = (
                "do_remote_decode", "do_remote_prefill", "ali_llumnix_disagg",
                "remote_host", "remote_port", "remote_engine_id",
                "remote_bootstrap_addr", "remote_block_ids", "remote_side_channel_port",
                "src_controller_address", "src_engine_id", "src_job_name", "uuid",
                "transfer_id",
            )
            if any((params or {}).get(key) for key in remote_fields):
                raise ValueError(
                    "Local Raiden simulation does not support remote PD handoff; "
                    "disable DashLLM remote decode routing and omit remote transfer params"
                )
            # Match the real producer's Prefill extent even without a Decode peer.
            # In-process simulation shares this list with frontend accounting.
            if request.prompt_token_ids is not None:
                request.prompt_token_ids = list(request.prompt_token_ids)
            result = original_admit(self, request)
            if self._sim_raiden_truncates_prompt:
                # CPU transport skips Stage-3 setup; preserve its native Prefill extent.
                self.connector_scheduler._maybe_truncate_prompt_for_handoff(request)
            return result

        def matched(self, request, num_computed_tokens):
            if not self._sim_local_raiden:
                return original_match(self, request, num_computed_tokens)
            # Raiden 不提供远端命中；MultiConnector 仍继续查询 V6D。
            return 0, False

        def allocated(self, request, blocks, num_external_tokens):
            if not self._sim_local_raiden:
                return original_alloc(self, request, blocks, num_external_tokens)
            if num_external_tokens != 0:
                raise ValueError("Local Raiden simulation cannot load remote PD tokens")
            # 不登记原生 stage3 register-ahead 或真实 send/load 任务。

        def metadata(self, scheduler_output):
            if not self._sim_local_raiden:
                return original_meta(self, scheduler_output)
            # 延迟导入：hook 安装时目标模块尚未完成类定义。
            from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
                TPUConnectorMetadata,
            )
            result = TPUConnectorMetadata()
            result.reqs_to_send = self._sim_raiden_pending_sends
            self._sim_raiden_pending_sends = {}
            return result

        def finished(self, request, block_ids):
            if not self._sim_local_raiden:
                return original_finish(self, request, block_ids)
            return False, None

        def finished_groups(self, request, block_ids):
            if not self._sim_local_raiden:
                return original_finish_groups(self, request, block_ids)
            if len(block_ids) != self._sim_raiden_num_groups:
                raise ValueError(
                    "Raiden simulation KV group count mismatch: "
                    f"expected {self._sim_raiden_num_groups}, got {len(block_ids)}"
                )
            from vllm.v1.request import RequestStatus

            req_id = request.request_id
            if getattr(request, "_sim_raiden_handoff_submitted", False):
                return req_id in self._sim_raiden_inflight, None
            if request.status not in (
                RequestStatus.FINISHED_LENGTH_CAPPED, RequestStatus.FINISHED_STOPPED
            ):
                return False, None
            tokens = self.connector_scheduler._stage3_transfer_tokens(
                request, request.num_computed_tokens
            )
            if tokens <= 0 or not any(block_ids):
                return False, None
            record = _LocalRaidenHandoff(
                tuple(tuple(ids) for ids in block_ids), tokens
            )
            self._sim_raiden_inflight[req_id] = record
            self._sim_raiden_pending_sends[req_id] = record
            request._sim_raiden_handoff_submitted = True
            # Native MultiConnector counts this independent asynchronous child.
            # Only its aggregated output permits scheduler block release.
            return True, None

        def update(self, output):
            if not self._sim_local_raiden:
                return original_update(self, output)
            for req_id in output.finished_sending or ():
                self._sim_raiden_inflight.pop(req_id, None)

        def pending(self):
            if not self._sim_local_raiden:
                return original_pending(self) if original_pending is not None else False
            worker = getattr(self, "connector_worker", None)
            return bool(
                self._sim_raiden_pending_sends or self._sim_raiden_inflight
                or (worker is not None and worker.has_pending_push_work())
            )

        target.__init__ = init
        target.on_new_request = admit
        target.get_num_new_matched_tokens = matched
        target.update_state_after_alloc = allocated
        target.build_connector_meta = metadata
        target.request_finished = finished
        target.request_finished_all_groups = finished_groups
        target.update_connector_output = update
        target.has_pending_push_work = pending
        target._sim_local_raiden_hook = True


def install_tpu_v6d_completion_guard(scheduler):
    """在已结算的 scheduler 回调中区分保存作业确认和请求级释放许可。"""
    if getattr(scheduler, "_sim_v6d_completion_guard", False):
        return
    connector = getattr(scheduler, "connector", None)
    if connector is None:
        return
    children = getattr(connector, "_connectors", (connector,))
    v6d_count = 0
    for child in children:
        child_type = type(child)
        if (
            child_type.__module__ == "kvs_connector.v6d_object_connector"
            and child_type.__name__ == "TPUV6dObjectConnector"
        ):
            v6d_count += 1
        elif (
            child_type.__module__ == "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
            and child_type.__name__ == "TPURaidenConnector"
            and getattr(child, "_sim_local_raiden", False)
        ):
            continue
        else:
            # 其他发送者有独立的 MultiConnector 完成计数，不能混用本适配。
            return
    if v6d_count != 1:
        return

    original_finished = scheduler._connector_finished
    original_update = connector.update_connector_output
    pending_release = set()

    def request_finished(request):
        if hasattr(request, "_sim_connector_finish_result"):
            return request._sim_connector_finish_result
        # 以原生返回值为准；仅请求终结且确实延迟释放才形成发送义务。
        result = original_finished(request)
        request._sim_connector_finish_result = result
        if result[0]:
            pending_release.add(request.request_id)
        else:
            pending_release.discard(request.request_id)
        return result

    def update_connector_output(output):
        incoming = set(output.finished_sending or ())
        live = incoming.intersection(scheduler.requests)
        # 已经释放的请求不能再次进入原生发布/清理，也不能再次释放。
        pending_release.difference_update(incoming - live)
        output.finished_sending = live or None
        try:
            # 先确认作业：中间 chunk 的缓存发布、内部状态清理仍归原生 V6D。
            original_update(output)
        except Exception:
            output.finished_sending = incoming or None
            raise
        # 此时已执行 update_from_output 的 token/终态更新，PP 早投递不会
        # 把 dispatch 时的状态误当成结算状态。不能用当步 finished_req_ids
        # 代替此处跨步保存的发送义务（它只是增量集合）。
        completed = {
            rid for rid in (output.finished_sending or ())
            if rid in pending_release
            and rid in scheduler.requests
            and scheduler.requests[rid].is_finished()
        }
        pending_release.difference_update(completed)
        output.finished_sending = completed or None
        # 不改变 finished_recving、worker metadata 或统计，不自行 free blocks。

    scheduler._connector_finished = request_finished
    connector.update_connector_output = update_connector_output
    scheduler._sim_v6d_completion_guard = True
    logger.info(
        "[V6D CPU] scheduler completion guard enabled: intermediate store acknowledgements "
        "do not release requests; final releases follow native delay_free"
    )


class _TPUConnectorWorkerCPUHook(BaseHook):
    """Shared lifecycle implementation for TPU v6d and Raiden workers."""

    @classmethod
    def _ensure_state(cls, worker):
        if not hasattr(worker, "_sim_tpu_load_done"):
            worker._sim_tpu_load_done = set()
        if not hasattr(worker, "_sim_tpu_store_done"):
            worker._sim_tpu_store_done = set()

    @classmethod
    def hook(cls, target):
        def start_load(self, *args, **kwargs):
            cls._ensure_state(self)
            metadata = _metadata_with(args, kwargs, "reqs_to_load")
            for req_id, groups in (getattr(metadata, "reqs_to_load", {}) or {}).items():
                # Empty key sets do not represent an external load.
                if any(len(keys) > 0 for keys, _ids in groups.values()):
                    self._sim_tpu_load_done.add(req_id)
            return None

        def start_store(self, *args, **kwargs):
            cls._ensure_state(self)
            metadata = _metadata_with(args, kwargs, "reqs_to_store")
            for req_id in (getattr(metadata, "reqs_to_store", {}) or {}):
                self._sim_tpu_store_done.add(req_id)
            return None

        def get_finished(self, *args, **kwargs):
            cls._ensure_state(self)
            loaded = set(self._sim_tpu_load_done)
            stored = set(self._sim_tpu_store_done)
            self._sim_tpu_load_done.clear()
            self._sim_tpu_store_done.clear()
            # vLLM/TPU contract: (finished_sending, finished_recving).
            return stored, loaded

        def has_pending_push_work(self):
            cls._ensure_state(self)
            return bool(self._sim_tpu_load_done or self._sim_tpu_store_done)

        def get_finished_count(self):
            cls._ensure_state(self)
            return len(self._sim_tpu_load_done) + len(self._sim_tpu_store_done)

        def register_kv_caches(self, kv_caches, *args, **kwargs):
            cls._ensure_state(self)
            self.kv_caches = kv_caches
            # Do not call native registration on fake 4-KiB KV tensors: the
            # native contract requires full storage views and byte assertions.
            return None

        def register_cross_layers_kv_cache(self, kv_caches, *args, **kwargs):
            cls._ensure_state(self)
            self.kv_caches = kv_caches
            return None

        def noop(self, *args, **kwargs):
            return None

        def empty_errors(self, *args, **kwargs):
            return []

        # The CPU adapter owns these queues even on workers without a native pending probe.
        target.has_pending_push_work = has_pending_push_work

        replacements = {
            "start_load_kv": start_load,
            "start_store_kv": start_store,
            "get_finished": get_finished,
            "get_finished_count": get_finished_count,
            "register_kv_caches": register_kv_caches,
            "register_cross_layers_kv_cache": register_cross_layers_kv_cache,
            "wait_for_jobs": noop,
            "save_kv_layer": noop,
            "handle_preemptions": noop,
            "get_block_ids_with_load_errors": empty_errors,
            "get_handshake_metadata": noop,
            "set_xfer_handshake_metadata": noop,
            "set_xfer_handshake_metadata_pp_aware": noop,
            "register_runner": noop,
            "shutdown": noop,
            "_register_v6d_host_memory": noop,
            "_async_v6d_init_and_notify": noop,
        }
        for name, replacement in replacements.items():
            if getattr(target, name, None) is not None:
                setattr(target, name, replacement)

        if getattr(target, "_start_async_v6d_init", None) is not None:
            def start_async_v6d_init(self):
                cls._ensure_state(self)
                self._load_handler = None
                self._store_handler = None
                return None

            target._start_async_v6d_init = start_async_v6d_init

        logger.info("[V6D Hijack] installed TPU CPU lifecycle hooks on %s", target)


class C_TPUV6dObjectConnectorWorkerHook(_TPUConnectorWorkerCPUHook):
    HOOK_MODULE_NAME = "kvs_connector.v6d_object_connector"
    HOOK_CLASS_NAME = "V6dObjectConnectorWorker"

    @classmethod
    def hook(cls, target):
        if target.__dict__.get("_sim_v6d_job_protocol", False):
            return
        original_get_finished = target.get_finished
        original_build_meta = getattr(target, "build_connector_worker_meta", None)
        super().hook(target)
        if original_build_meta is None:
            # 旧的请求级协议继续使用既有 adapter，不给 Raiden 改协议。
            return
        cpu_load = target.start_load_kv

        def queue_for(self):
            if not hasattr(self, "_sim_tpu_store_queue"):
                self._sim_tpu_store_queue = _TPUStoreQueue()
            return self._sim_tpu_store_queue

        def validate(self, metadata):
            if not hasattr(metadata, "store_job_ids") or not hasattr(metadata, "store_waits"):
                raise ValueError("TPU V6D job protocol requires store_job_ids and store_waits")
            if metadata.num_groups != self._num_groups:
                raise ValueError("TPU V6D metadata/worker KV group count mismatch")
            if self._load_handler is not None or self._store_handler is not None:
                raise RuntimeError("TPU V6D CPU job adapter requires disabled DMA handlers")
            owned = {}
            for job in queue_for(self).jobs:
                owned.setdefault(job.req_id, set()).add(job.job_id)
            saving = {rid: jobs for rid, jobs in self._saving_req_jobs.items() if jobs}
            if (self._pending_jobs or saving != owned or self._loading_req_jobs
                    or getattr(self, "_store_job_ids", {})):
                raise RuntimeError("TPU V6D CPU job adapter found unexpected native DMA jobs")

        def load_queue_for(self):
            if not hasattr(self, "_sim_tpu_load_queue"):
                self._sim_tpu_load_queue = _TimedReqQueue(
                    lanes=_positive_env_int("HISIM_TPU_V6D_LOAD_LANES", 1)
                )
            return self._sim_tpu_load_queue

        def start_load(self, metadata):
            validate(self, metadata)
            read_bps = _tpu_load_bytes_per_second()
            if read_bps > 0:
                cls._ensure_state(self)
                if not hasattr(self, "_sim_tpu_node_layout"):
                    self._sim_tpu_node_layout = _tpu_node_object_layout(
                        self._vllm_config, self._kv_cache_config
                    )
                layout = self._sim_tpu_node_layout
                if layout is not None:
                    now = time.monotonic()
                    lqueue = load_queue_for(self)
                    for req_id, groups in (getattr(metadata, "reqs_to_load", {}) or {}).items():
                        n_objs = sum(len(keys) for keys, _ids in groups.values())
                        if n_objs > 0:
                            lqueue.enqueue(req_id, n_objs * layout[0] / read_bps, now)
                else:
                    cpu_load(self, metadata)
            else:
                cpu_load(self, metadata)
            # 这是原生 scheduler 明确承诺 delay_free 的增量集合，可能只有
            # wait 而没有新 store/load；不能拿 finished_req_ids 代替它。
            self._finished_reqs_waiting_for_store.update(metadata.store_waits)

        def start_store(self, metadata, nan_check=None):
            validate(self, metadata)
            if nan_check is not None:
                raise ValueError("TPU V6D CPU simulation cannot validate real NaN snapshots")
            stores = metadata.reqs_to_store
            if set(stores) != set(metadata.store_job_ids):
                raise ValueError("TPU V6D store_job_ids must match reqs_to_store")
            queue = queue_for(self)
            for req_id, groups in stores.items():
                job_id = metadata.store_job_ids[req_id]
                if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id < 0:
                    raise ValueError("TPU V6D store job ID must be a non-negative integer")
                for keys, blocks, masks in groups.values():
                    if not len(keys) == len(blocks) == len(masks):
                        raise ValueError("TPU V6D store keys/blocks/node masks length mismatch")
                if job_id in queue.seen:
                    if queue.seen[job_id] != req_id:
                        raise ValueError("Conflicting TPU V6D store job ID")
                    continue
                # Native build_connector_meta allocates IDs in dispatch order.
                if job_id <= queue.highest_job_id:
                    continue
                if not hasattr(self, "_sim_tpu_node_layout"):
                    self._sim_tpu_node_layout = _tpu_node_object_layout(
                        self._vllm_config, self._kv_cache_config
                    )
                layout = self._sim_tpu_node_layout
                selected = sum(
                    bool(mask & (1 << self._node_id))
                    for _keys, _blocks, masks in groups.values() for mask in masks
                ) if layout is not None else 0
                if selected:
                    duration = selected * layout[0] / _tpu_store_bytes_per_second()
                    gate = _transfer_gate.get()
                    if gate is None:
                        gate = _TPUTransferGate(time.monotonic())
                    queue.jobs.append(_TPUStoreJob(job_id, req_id, gate, duration))
                    self._saving_req_jobs.setdefault(req_id, set()).add(job_id)
                    queue.seen[job_id] = req_id
                else:
                    # No DMA for mask=0; do not wait on PP or occupy the lane.
                    # Non-TPU/non-PP keeps its existing immediate ACK behavior.
                    acknowledge(self, job_id)
                queue.highest_job_id = job_id
            # No synthetic DMA entry in _pending_jobs, and no seal/free here.

        def acknowledge(self, job_id):
            ranks = self._worker_meta.store_results.setdefault(job_id, {})
            if self._worker_rank in ranks and ranks[self._worker_rank] is not None:
                raise ValueError("Conflicting TPU V6D store result")
            ranks[self._worker_rank] = None

        def harvest(self):
            queue = queue_for(self)
            queue.resolve_deadlines()
            now = time.monotonic()
            if queue._num_lanes == 1:
                while queue.jobs:
                    job = queue.jobs[0]
                    if job.deadline is None or job.deadline > now:
                        break
                    acknowledge(self, job.job_id)
                    self._saving_req_jobs[job.req_id].remove(job.job_id)
                    if not self._saving_req_jobs[job.req_id]:
                        del self._saving_req_jobs[job.req_id]
                    queue.jobs.popleft()
                    del queue.seen[job.job_id]
            else:
                remaining = deque()
                for job in queue.jobs:
                    if job.deadline is not None and job.deadline <= now:
                        acknowledge(self, job.job_id)
                        self._saving_req_jobs[job.req_id].remove(job.job_id)
                        if not self._saving_req_jobs[job.req_id]:
                            del self._saving_req_jobs[job.req_id]
                        del queue.seen[job.job_id]
                    else:
                        remaining.append(job)
                queue.jobs = remaining
            if hasattr(self, "_sim_tpu_load_queue"):
                self._sim_tpu_load_done.update(self._sim_tpu_load_queue.harvest(now))

        def handle_preemptions(self, metadata):
            queue = queue_for(self)
            preempted = set(metadata.preempted_req_ids or ())
            selected = [i for i, job in enumerate(queue.jobs) if job.req_id in preempted]
            if not selected:
                return
            jobs = list(queue.jobs)[:max(selected) + 1]
            for job in jobs:
                if job.gate.ready_at is None:
                    if job.gate.wait_ready is None:
                        raise RuntimeError("Preempted store has no PP completion future")
                    job.gate.wait_ready()
            queue.resolve_deadlines()
            delay = max(jobs[i].deadline for i in selected) - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        def get_finished(self, finished_req_ids=None):
            cls._ensure_state(self)
            harvest(self)
            # 原生实现仅在 store_waits 已登记且没有未完成 DMA 时返回发送终态。
            # handler 为 None，不会访问 TPU；builder 稍后独立导出作业确认。
            sent, received = original_get_finished(self, finished_req_ids or set())
            received = set(received or ()) | self._sim_tpu_load_done
            self._sim_tpu_load_done.clear()
            return sent, received

        def has_pending_push_work(self):
            cls._ensure_state(self)
            load_pending = (
                bool(self._sim_tpu_load_queue.jobs)
                if hasattr(self, "_sim_tpu_load_queue")
                else False
            )
            return bool(
                queue_for(self).jobs or load_pending or self._sim_tpu_load_done
                or self._finished_reqs_waiting_for_store
                or self._worker_meta.store_results or self._worker_meta.failed_loads
            )

        target.start_load_kv = start_load
        target.start_store_kv = start_store
        target.handle_preemptions = handle_preemptions
        target.get_finished = get_finished
        target.has_pending_push_work = has_pending_push_work
        # 不重写原生 builder：保持 V6dWorkerMetadata 类型、聚合和换对象语义，
        # 已投递的 PP 输出不能被下一步清空。
        target.build_connector_worker_meta = original_build_meta
        target._sim_v6d_job_protocol = True
        logger.info(
            "[V6D CPU] job-result protocol enabled: store_results by job/rank; "
            "request terminals follow store_waits"
        )


class C_TPURaidenConnectorWorkerHook(_TPUConnectorWorkerCPUHook):
    HOOK_MODULE_NAME = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    HOOK_CLASS_NAME = "TPURaidenConnectorWorker"

    @classmethod
    def hook(cls, target):
        super().hook(target)
        base_get_finished = target.get_finished
        base_has_pending = target.has_pending_push_work

        def hold_queue_for(self):
            if not hasattr(self, "_sim_raiden_hold_queue"):
                self._sim_raiden_hold_queue = _TimedReqQueue(
                    lanes=_positive_env_int("HISIM_TPU_RAIDEN_HANDOFF_LANES", 4)
                )
            return self._sim_raiden_hold_queue

        def process_send_load(
            self, metadata, wait_for_completion=False, report_completion=True
        ):
            cls._ensure_state(self)
            if self.is_producer:
                base_s, bw_bps = _tpu_raiden_handoff_timing()
                if base_s > 0 or bw_bps > 0:
                    now = time.monotonic()
                    hqueue = hold_queue_for(self)
                    bytes_per_token = _nonneg_env_float(
                        "HISIM_TPU_RAIDEN_BYTES_PER_TOKEN", 0.0
                    )
                    for req_id, req_meta in (getattr(metadata, "reqs_to_send", {}) or {}).items():
                        tokens = max(0, int(getattr(req_meta, "num_tokens", 0) or 0))
                        xfer_s = (
                            (tokens * bytes_per_token / bw_bps)
                            if (bw_bps > 0 and bytes_per_token > 0 and tokens > 0)
                            else 0.0
                        )
                        duration = base_s + xfer_s
                        if duration > 0:
                            hqueue.enqueue(req_id, duration, now)
                        else:
                            self._sim_tpu_store_done.add(req_id)
                else:
                    for req_id in (getattr(metadata, "reqs_to_send", {}) or {}):
                        self._sim_tpu_store_done.add(req_id)
                return None
            if not report_completion:
                return None
            for req_id, req_meta in (getattr(metadata, "reqs_to_load", {}) or {}).items():
                if getattr(req_meta, "report_completion", True):
                    self._sim_tpu_load_done.add(req_id)
            return None

        def get_finished(self, *args, **kwargs):
            cls._ensure_state(self)
            if hasattr(self, "_sim_raiden_hold_queue"):
                self._sim_tpu_store_done.update(
                    self._sim_raiden_hold_queue.harvest(time.monotonic())
                )
            return base_get_finished(self, *args, **kwargs)

        def has_pending_push_work(self):
            cls._ensure_state(self)
            if hasattr(self, "_sim_raiden_hold_queue") and self._sim_raiden_hold_queue.jobs:
                return True
            return base_has_pending(self)

        target.process_send_load = process_send_load
        target.get_finished = get_finished
        target.has_pending_push_work = has_pending_push_work
