"""Coordinate request replay, inference timing, and statistics.

EngineCore parks OFFLINE arrivals before native cache admission. Scheduler
counts keep parked arrivals visible to the engine loop. The worker's model
runner predicts each step from its DP-wide batch; the executor then defers
completion on the wall clock (BLOCKING) or advances virtual time (OFFLINE).
The hooks share mode and iteration state on EngineCoreHook.
"""

import heapq
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict

from hisim.hook import BaseHook
from hisim.simulation.manager import StateManager
from hisim.env import Envs
from hisim.simulation.req_stats_manager import request_stats_manager
from hisim.simulation.types import SimulationMode
from hisim.time_predictor import ScheduleBatch
from hisim.time_predictor import ScheduleRequest
from hisim.utils import get_logger

logger = get_logger()

# Snapshot overrides once; None preserves each request's token limit.
_MAX_TOKENS = Envs.max_tokens()

# BLOCKING statistics are opt-in to avoid hot-path collection overhead.
_RECORD_STATS = Envs.record_stats()


def _stats_recording_enabled() -> bool:
    # OFFLINE metrics require token timestamps regardless of the stats flag.
    return C_VLLMEngineCoreHook.SIM_MODE != SimulationMode.BLOCKING or _RECORD_STATS


def _new_request_stats(request, created_time, queue_start, last_event_time):
    """Initialize one RequestStats entry when stats collection is enabled."""
    if not _stats_recording_enabled():
        return None

    input_length = getattr(request, "num_prompt_tokens", None)
    if input_length is None:
        input_length = len(getattr(request, "prompt_token_ids", None) or [])
    st = request_stats_manager.get_req_stats(request.request_id)
    st.created_time = created_time
    st.queue_start = queue_start
    st.queue_end = -1
    st.gen_token_latencies = []
    st.last_event_time = last_event_time
    st.input_length = input_length
    st.output_length = 0
    return st


class ReqDispatcher:
    """Park OFFLINE arrivals until virtual time permits native admission."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        # Sequence numbers break equal-time ties without comparing requests.
        self.future_queue: list[tuple[float, int, object, object, int]] = []
        self.seq_counter = 0
        self.total_expected = float("inf")
        self.all_received = False
        self.req_created_time: dict[str, float] = {}
        # Capture the native admission method when installing the engine hook.
        self.original_core_add_request = None
        self.__class__._initialized = True

    def reset(self):
        self.future_queue.clear()
        self.seq_counter = 0
        self.total_expected = float("inf")
        self.all_received = False
        self.req_created_time.clear()

    def __len__(self) -> int:
        return len(self.future_queue)

    def has_next(self) -> bool:
        return len(self.future_queue) > 0

    def next_req_created_time(self) -> float:
        return self.future_queue[0][0]

    def add(self, engine_core, request, request_wave: int, created_time: float):
        """Park a request fully inert until its created_time."""
        heapq.heappush(
            self.future_queue,
            (created_time, self.seq_counter, engine_core, request, request_wave),
        )
        self.req_created_time[request.request_id] = created_time
        self.seq_counter += 1

        if self.seq_counter >= self.total_expected:
            self.all_received = True
            logger.info(
                "All %d requests received. Starting simulation.",
                self.total_expected,
            )

    def dispatch(self, engine_core):
        """Admit due arrivals through the native engine/cache path."""
        current_time = StateManager.get_global_clock()
        while self.future_queue and self.future_queue[0][0] <= current_time:
            _, _, core, request, request_wave = heapq.heappop(self.future_queue)
            ct = self.req_created_time.get(request.request_id, current_time)
            _new_request_stats(
                request,
                created_time=ct,
                queue_start=current_time,
                last_event_time=ct,
            )
            self.original_core_add_request(core, request, request_wave)


class C_VLLMSchedulerHook(BaseHook):
    """Count parked arrivals so the engine loop does not go idle before dispatch."""

    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target):
        original_get_num_unfinished = target.get_num_unfinished_requests
        target.get_num_unfinished_requests = lambda self: (
            original_get_num_unfinished(self) + len(ReqDispatcher())
        )
        original_has_unfinished = getattr(target, "has_unfinished_requests", None)
        if original_has_unfinished is not None:
            target.has_unfinished_requests = lambda self: (
                original_has_unfinished(self) or len(ReqDispatcher()) > 0
            )


def _ext_computed_tokens(req_id) -> int:
    """Read external-hit tokens from the live connector request, or return zero."""
    try:
        from vllm.v1.hybrid_connector.engine_proxy import _sched

        req = _sched().requests.get(req_id)
        return max(getattr(req, "num_external_computed_tokens", 0) or 0, 0)
    except Exception:
        return 0


def _finalize_step_stats(scheduler_output, event_time: float) -> None:
    """Publish one step's stats when its model-output future is complete."""
    if not _stats_recording_enabled():
        return
    if getattr(scheduler_output, "_sim_stats_finalized", False):
        return
    iteration_stat = getattr(scheduler_output, "_sim_iteration_stat", None)
    if iteration_stat is None:
        return
    scheduler_output._sim_stats_finalized = True
    C_VLLMEngineCoreHook.ITERATION_STATS.append(iteration_stat)

    token_emitted = getattr(scheduler_output, "_sim_token_emitted", {})
    for req_id in getattr(scheduler_output, "num_scheduled_tokens", None) or {}:
        if not token_emitted.get(req_id, True):
            continue
        st = request_stats_manager.stats.get(req_id)
        if st is not None:
            st.gen_token_latencies.append(event_time - st.last_event_time)
            st.last_event_time = event_time


class C_VLLMEngineCoreHook(BaseHook):
    """Gate OFFLINE admission and export per-round stats.

    Parking before native add_request prevents early connector lookup and KV
    allocation. Input preprocessing remains native; BLOCKING admits immediately.
    """

    HOOK_CLASS_NAME = "EngineCore"
    HOOK_MODULE_NAME = "vllm.v1.engine.core"

    SIM_MODE: SimulationMode = SimulationMode(Envs.simulation_mode())

    # Publish iteration stats after wall-clock or virtual-clock completion.
    ITERATION_STATS: list[dict] = []

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_add_request = target.add_request
        original_step = target.step
        original_wait_model_output = getattr(target, "_wait_model_output_future", None)
        dispatcher = ReqDispatcher()
        dispatcher.original_core_add_request = original_add_request

        def wrapped_init(self, vllm_config, *args, **kwargs):
            """Reset replay/statistics before the native engine starts."""
            ReqDispatcher().reset()
            request_stats_manager.reset()
            cls.ITERATION_STATS.clear()

            original_init(self, vllm_config, *args, **kwargs)

        def wrapped_add_request(self, request, request_wave=0, *args, **kwargs):
            """Queue timestamped OFFLINE requests; admit all others immediately."""
            if _MAX_TOKENS is not None:
                if request.sampling_params is not None:
                    request.sampling_params.max_tokens = _MAX_TOKENS
                    request.sampling_params.ignore_eos = True
                request.max_tokens = _MAX_TOKENS

            created_time = None
            sp = request.sampling_params
            if sp is not None and sp.extra_args:
                sim = sp.extra_args.get("simulation")
                if sim:
                    created_time = sim.get("created_time")
                    total = sim.get("total_request")
                    if total is not None:
                        dispatcher.total_expected = total

            if created_time is None or cls_sim_mode() == SimulationMode.BLOCKING:
                if cls_sim_mode() == SimulationMode.BLOCKING:
                    # Record admission before the connector can consume the request.
                    now = time.time()
                    _new_request_stats(
                        request,
                        created_time=created_time if created_time is not None else now,
                        queue_start=now,
                        last_event_time=now,
                    )
                original_add_request(self, request, request_wave, *args, **kwargs)
                return

            dispatcher.add(self, request, request_wave, created_time)

        def wrapped_step(self, *args, **kwargs):
            """Dispatch due future-queue requests before each engine step."""
            offline = cls_sim_mode() == SimulationMode.OFFLINE
            if offline and dispatcher.all_received:
                dispatcher.dispatch(self)

            result = original_step(self, *args, **kwargs)
            if offline and not result[1]:
                # The count hook includes parked arrivals; they are not active work.
                is_req_pending = self.scheduler.get_num_unfinished_requests() > len(
                    dispatcher
                ) or _connector_has_pending(self.scheduler)
                if is_req_pending:
                    # Tick while native/connector work remains; otherwise jump to arrival.
                    StateManager.step_global_clock(0.005)
                    StateManager.set_current_inference_dur(0.005)
                elif dispatcher.all_received and dispatcher.has_next():
                    StateManager.set_global_clock(
                        max(
                            StateManager.get_global_clock(),
                            dispatcher.next_req_created_time(),
                        )
                    )
            return result

        def wrapped_wait_model_output(
            self, model_output_future, step_sout, *args, **kwargs
        ):
            result = original_wait_model_output(
                self, model_output_future, step_sout, *args, **kwargs
            )
            if cls.SIM_MODE == SimulationMode.BLOCKING:
                # Use the resolved output's completion time, not a projected clock.
                _finalize_step_stats(step_sout, time.time())
            return result

        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.step = wrapped_step
        if original_wait_model_output is not None:
            target._wait_model_output_future = wrapped_wait_model_output

        # Profile marks round boundaries: export/reset stats, not caches.
        # Keep this here because class registration applies only the first match.
        def wrapped_profile(self, is_start: bool = True):
            req_stats = request_stats_manager.get_all_req_stats()
            output_dir = Envs.output_dir()
            parallel = self.vllm_config.parallel_config
            if parallel.data_parallel_size > 1:
                # DP ranks are separate engine processes sharing the output dir.
                output_dir = os.path.join(
                    output_dir, f"dp{parallel.data_parallel_rank}"
                )
                os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "request.jsonl"), "w") as f:
                for item in req_stats:
                    f.write(json.dumps(asdict(item), default=str) + "\n")
            with open(os.path.join(output_dir, "iteration.jsonl"), "w") as f:
                for item in cls.ITERATION_STATS:
                    f.write(json.dumps(item, default=str) + "\n")

            logger.info(
                "[ProfileHook] Exported %d requests, %d iterations to %s",
                len(req_stats),
                len(cls.ITERATION_STATS),
                output_dir,
            )

            request_stats_manager.reset()
            cls.ITERATION_STATS.clear()
            # Start a fresh replay epoch in the engine-core process.
            StateManager.reset()
            ReqDispatcher().reset()

        target.profile = wrapped_profile


def cls_sim_mode() -> SimulationMode:
    return C_VLLMEngineCoreHook.SIM_MODE


def _connector_has_pending(scheduler) -> bool:
    """Check native work interfaces, including loads and trailing stores."""
    get_conn = getattr(scheduler, "get_kv_connector", None)
    connector = get_conn() if get_conn is not None else None
    if connector is None:
        return False
    return bool(
        connector.has_requests() or getattr(connector, "pending_requests", None)
    )


class C_VLLMExecutorHook(BaseHook):
    """Annotate SchedulerOutput and defer or virtually advance completion.

    The worker attaches the predicted span (`_sim_step`). Async scheduling
    consumes it in worker output; non-async scheduling and DP dummy steps
    return a deferred executor future. Both leave native connector polling
    available while BLOCKING inference is pending.
    """

    HOOK_CLASS_NAME = "UniProcExecutor"
    HOOK_MODULE_NAME = r"vllm(\.v1)?\.executor\.uniproc_executor"
    REGEX = True

    # Serialize deferred completions off the engine thread, like one GPU stream.
    _GPU_SPAN_EXECUTOR = None

    @classmethod
    def _gpu_span_pool(cls):
        if cls._GPU_SPAN_EXECUTOR is None:
            cls._GPU_SPAN_EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="sim-gpu-span"
            )
        return cls._GPU_SPAN_EXECUTOR

    @classmethod
    def hook(cls, target):
        original_execute_model = target.execute_model
        original_execute_dummy_batch = target.execute_dummy_batch

        # Retain prompt length/logprobs across cached steps; discard on finish.
        req_info: dict[str, tuple] = {}

        def wrapped_execute_model(self, scheduler_output, *args, **kwargs):
            num_scheduled_tokens = getattr(
                scheduler_output, "num_scheduled_tokens", None
            )
            if not num_scheduled_tokens:
                return original_execute_model(self, scheduler_output, *args, **kwargs)

            # Queue delay ends at scheduling, before accounting for this step.
            queue_end_time = (
                time.time()
                if C_VLLMEngineCoreHook.SIM_MODE == SimulationMode.BLOCKING
                else StateManager.get_global_clock()
            )
            # At admission, split computed tokens into local and external hits.
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or ()
            for new_req_data in new_reqs:
                req_id = new_req_data.req_id
                st = request_stats_manager.stats.get(req_id)
                prompt_ids = getattr(new_req_data, "prompt_token_ids", None)
                sp = getattr(new_req_data, "sampling_params", None)

                # Queue/hit stats, recorded once at the first schedule
                if st is not None and st.queue_end == -1:
                    st.queue_end = queue_end_time
                    st.input_length = len(prompt_ids) if prompt_ids is not None else 0
                    st.output_length = getattr(sp, "max_tokens", 0) or 0
                    cached = max(
                        getattr(new_req_data, "num_computed_tokens", 0) or 0, 0
                    )
                    ext = _ext_computed_tokens(req_id)
                    st.final_device_hit_len = cached
                    st.ext_kv_hit_len = ext
                    st.local_kv_hit_len = cached - ext
                    # The shared metrics schema maps host hits to external tokens.
                    st.final_host_hit_len = ext
                    # Log the admission snapshot once.
                    logger.info(
                        "[HitRate] rid=%s input_len=%d local_hit=%d ext_hit=%d",
                        st.rid,
                        st.input_length,
                        st.local_kv_hit_len,
                        st.ext_kv_hit_len,
                    )

                req_info[req_id] = (
                    len(prompt_ids) if prompt_ids is not None else None,
                    bool(getattr(sp, "logprobs", None)),
                )
            for fin_id in getattr(scheduler_output, "finished_req_ids", None) or ():
                req_info.pop(fin_id, None)

            # SchedulerOutput records pre-step KV lengths for new and cached requests.
            num_computed = {r.req_id: r.num_computed_tokens for r in new_reqs}
            cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if cached is not None:
                num_computed.update(zip(cached.req_ids, cached.num_computed_tokens))

            # Emit only after the full prompt; pass flags to the mock worker output.
            simulation_batch = ScheduleBatch(reqs=[])
            token_emitted: dict[str, bool] = {}
            logprobs_tokens = 0
            for req_id, num_tokens in num_scheduled_tokens.items():
                if req_id not in num_computed:
                    continue
                past_kv_length = num_computed[req_id]
                prompt_tokens, has_logprobs = req_info.get(req_id, (None, False))
                token_emitted[req_id] = (
                    prompt_tokens is None
                    or past_kv_length + num_tokens >= prompt_tokens
                )
                if has_logprobs:
                    logprobs_tokens += num_tokens
                simulation_batch.reqs.append(
                    ScheduleRequest(
                        extend_length=num_tokens,
                        past_kv_length=max(0, past_kv_length),
                    )
                )
            scheduler_output._sim_token_emitted = token_emitted
            scheduler_output._sim_logprobs_tokens = logprobs_tokens

            if simulation_batch.is_empty():
                return original_execute_model(self, scheduler_output, *args, **kwargs)

            StateManager.inc_iteration()
            # The worker coordinates the DP-wide batch, then predicts the span.
            result = original_execute_model(self, scheduler_output, *args, **kwargs)
            if isinstance(result, Future) and result.exception() is not None:
                return result
            step = getattr(scheduler_output, "_sim_step", None)
            if step is None:
                raise RuntimeError(
                    "scheduler_output lacks the worker's _sim_step annotation "
                    "(sim worker hook not active?)"
                )
            full_step_latency = step.full_step_latency

            # Publish these annotations only when the modeled step completes.
            scheduler_output._sim_iteration_stat = {
                "requests": step.requests,
                # RPC-1 only, matching the predictor's training label.
                "forward_latency": step.forward_latency,
                "sample_tokens_latency": step.sample_tokens_latency,
                "logprobs_latency": step.logprobs_latency,
                "full_step_latency": full_step_latency,
                "prediction_cpu_latency": step.prediction_cpu_latency,
                "blocking_wait_latency": step.blocking_wait_latency,
                "total_tokens": sum(num_scheduled_tokens.values()),
                "l2_load_latency": 0.0,
                "l2_backup_latency": 0.0,
                **step.dp_fields(),
            }

            if C_VLLMEngineCoreHook.SIM_MODE != SimulationMode.BLOCKING:
                # Virtual time includes modeled costs, not real engine/IPC overhead.
                StateManager.set_current_inference_dur(full_step_latency)
                StateManager.step_global_clock(full_step_latency)
                _finalize_step_stats(scheduler_output, StateManager.get_global_clock())
                return result

            if _async_scheduling(self):
                # Async scheduling consumes the span in the worker's sample output.
                return result

            # Non-async scheduling delays the mock output's future.
            # The native bypass loop can still process input and connector work.
            return cls._defer(result, step.blocking_wait_latency, scheduler_output)

        def wrapped_execute_dummy_batch(self, *args, non_block=False, **kwargs):
            """Hold an idle DP rank's dummy step for the coordinated span."""
            result = original_execute_dummy_batch(
                self, *args, non_block=non_block, **kwargs
            )
            if C_VLLMEngineCoreHook.SIM_MODE != SimulationMode.BLOCKING or (
                isinstance(result, Future) and result.exception() is not None
            ):
                return result
            model_output = result.result() if isinstance(result, Future) else result
            step = model_output.__dict__.pop("_sim_step")
            deferred = cls._defer(result, step.blocking_wait_latency, None)
            return deferred if non_block else deferred.result()

        target.execute_model = wrapped_execute_model
        target.execute_dummy_batch = wrapped_execute_dummy_batch
        logger.info(
            "[vLLM Hijack] UniProcExecutor hook installed "
            "(DP-coordinated prediction in worker; BLOCKING span deferred)"
        )

    @classmethod
    def _defer(cls, result, span: float, scheduler_output) -> Future:
        """Resolve the model output after the span, serialized like a GPU stream."""
        model_output = result.result() if isinstance(result, Future) else result
        deferred: Future = Future()

        def _consume_gpu_span():
            try:
                time.sleep(span)
                # Finalize before resolving the future and admitting the next step.
                if scheduler_output is not None:
                    _finalize_step_stats(scheduler_output, time.time())
                deferred.set_result(model_output)
            except BaseException as exc:  # pragma: no cover
                deferred.set_exception(exc)

        cls._gpu_span_pool().submit(_consume_gpu_span)
        return deferred


def _async_scheduling(executor) -> bool:
    scheduler_config = getattr(executor, "scheduler_config", None)
    return bool(getattr(scheduler_config, "async_scheduling", False))
