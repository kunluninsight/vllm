"""Simulate vLLM worker execution with a CPU model runner.

Discover native KV specs with a meta-device model walk, but allocate
at most 4 KiB per CPU backing tensor. v6d object sizes follow the specs, not
these buffers. Mock token output retains the connector lifecycle; step timing
comes from `SimModelRunner`.
"""

import asyncio
import dataclasses
import threading
import time

import torch

from hisim.hook import BaseHook
from hisim.simulation.manager import ConfigManager
from hisim.simulation.types import SimulationMode
from hisim.simulation.utils import profile_device_available_bytes
from hisim.simulation.hijack.vllm.engine_core_pipeline import C_VLLMEngineCoreHook
from hisim.simulation.hijack.vllm.model_runner import SimModelRunner
from hisim.utils import get_logger

logger = get_logger()


def _local_requests(scheduler_output) -> list[tuple[int, int, int]]:
    """Collector-compatible `(extend, past_kv, output_len)` rows of this rank."""
    new_reqs = scheduler_output.scheduled_new_reqs or ()
    past = {r.req_id: r.num_computed_tokens for r in new_reqs}
    output_len = dict.fromkeys(past, 0)
    cached = scheduler_output.scheduled_cached_reqs
    if cached is not None:
        past.update(zip(cached.req_ids, cached.num_computed_tokens))
        num_output = getattr(cached, "num_output_tokens", None)
        if num_output:
            output_len.update(zip(cached.req_ids, num_output))
    return [
        (int(num_tokens), max(0, int(past[req_id])), int(output_len.get(req_id, 0)))
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items()
        if req_id in past
    ]


def _build_kv_cache_spec(vllm_config) -> dict:
    """Discover native layer metadata without loading weights or payload buffers."""
    from hisim.simulation.hijack.vllm.kv_layout import build_kv_cache_spec

    return build_kv_cache_spec(vllm_config)


class C_VLLMWorkerHook(BaseHook):
    """Replace GPU execution with CPU placeholders and native connector calls."""

    HOOK_CLASS_NAME = "Worker"
    HOOK_MODULE_NAME = r"vllm\.v1\.worker\.(gpu_worker|worker)"
    REGEX = True

    @classmethod
    def hook(cls, target):
        from vllm.v1.outputs import (
            AsyncModelRunnerOutput as _AsyncModelRunnerOutput,
            ModelRunnerOutput as _ModelRunnerOutput,
        )

        class _SimAsyncOutput(_AsyncModelRunnerOutput):
            """Defer a simulated GPU span to vLLM's async-output thread."""

            def __init__(self, output, latency: float):
                self._output = output
                self._latency = latency

            def get_output(self):
                time.sleep(self._latency)
                return self._output

        # Field set varies across vLLM versions; probe once at install time.
        _mro_fields = {f.name for f in dataclasses.fields(_ModelRunnerOutput)}

        _KVConnectorOutput = None
        try:
            from vllm.v1.outputs import KVConnectorOutput as _KVConnectorOutput
        except ImportError:
            pass

        _has_kv_transfer_group = None
        _get_kv_transfer_group = None
        try:
            from vllm.distributed.kv_transfer import (
                has_kv_transfer_group as _has_kv_transfer_group,
                get_kv_transfer_group as _get_kv_transfer_group,
            )
        except ImportError:
            pass

        def override_init_device(self):
            """Initialize Gloo, CPU workspace, and the CPU model runner."""
            try:
                from vllm.v1.worker.gpu_worker import (
                    init_worker_distributed_environment,
                    init_workspace_manager,
                    set_random_seed,
                )
            except ImportError:
                from vllm.v1.worker.worker import (
                    init_worker_distributed_environment,
                    init_workspace_manager,
                    set_random_seed,
                )

            # Supply the config context required by the distributed initializer.
            try:
                from vllm.config.vllm import set_current_vllm_config

                with set_current_vllm_config(self.vllm_config):
                    init_worker_distributed_environment(
                        self.vllm_config,
                        self.rank,
                        self.distributed_init_method,
                        self.local_rank,
                        "gloo",
                    )
            except (ImportError, TypeError):
                # Compatibility path without the config context helper.
                init_worker_distributed_environment(
                    self.vllm_config,
                    self.rank,
                    self.distributed_init_method,
                    self.local_rank,
                    "gloo",
                )
            set_random_seed(self.vllm_config.model_config.seed)

            self.device = torch.device("cpu")
            self.model_runner = SimModelRunner(
                self.vllm_config, kv_spec=_build_kv_cache_spec(self.vllm_config)
            )

            self.init_snapshot = None
            self.requested_memory = 0
            init_workspace_manager(self.device, 1)
            logger.info(
                "[vLLM Hijack] Worker.init_device: CPU model runner initialized"
            )

        def override_load_model(self, *args, **kwargs):
            """Load the latency predictor in place of model weights."""
            self.model_runner.load_model()

        def override_determine_available_memory(self):
            """Profile modeled KV capacity, with an 80 GiB fallback on failure."""
            try:
                available_bytes = profile_device_available_bytes(
                    self.model_runner.model_info,
                    ConfigManager.get_accelerator_info(),
                    self.model_runner.scheduler_config,
                )

                logger.info(
                    "[vLLM Hijack] Worker.determine_available_memory: %d gibibytes. The available memory will be used for kv cache allocation.",
                    available_bytes // (1 << 30),
                )

                return available_bytes
            except Exception as e:
                # Profiling needs the optional aiconfigurator extra; without it
                # the fallback silently distorts the KV cache block count.
                logger.warning(
                    "[vLLM Hijack] Worker.determine_available_memory: profiling "
                    "failed (%s), falling back to 80 gibibytes. Install "
                    "hisim[aiconfigurator] for modeled KV capacity.",
                    e,
                )
                return 80 * (1 << 30)  # 80 GiB fallback

        def override_get_kv_cache_spec(self):
            """Return pre-built KV cache spec."""
            return self.model_runner.get_kv_cache_spec()

        def override_initialize_from_config(self, kv_cache_config):
            """Init KV connector and allocate CPU KV cache tensors."""
            from vllm.distributed.kv_transfer import (
                ensure_kv_transfer_initialized,
                has_kv_transfer_group,
                get_kv_transfer_group,
            )

            # Use the first shard when the engine supplies a config list.
            if isinstance(kv_cache_config, (list, tuple)):
                kv_cache_config = kv_cache_config[0]

            self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
            num_blocks = kv_cache_config.num_blocks

            # Provide the connector loop without GPU worker initialization.
            # Temporarily replace worker_init only during connector construction.
            try:
                import vllm.v1.hybrid_connector.engine_proxy as _engine_proxy

                _orig_worker_init = _engine_proxy.worker_init

                def _ensure_cpu_worker_loop():
                    if getattr(_engine_proxy, "_g_worker_loop", None) is not None:
                        return
                    loop = asyncio.new_event_loop()

                    def _run_loop():
                        asyncio.set_event_loop(loop)
                        loop.run_forever()

                    thread = threading.Thread(
                        target=_run_loop,
                        name="hybridworker-cpu-loop",
                        daemon=True,
                    )
                    thread.start()
                    _engine_proxy._g_worker_loop = loop
                    logger.info("[vLLM Hijack] installed CPU hybrid worker loop")

                def _cpu_worker_init(vllm_config, local_rank):
                    _ensure_cpu_worker_loop()
                    return None

                _ensure_cpu_worker_loop()
                _engine_proxy.worker_init = _cpu_worker_init
            except (ImportError, AttributeError):
                _orig_worker_init = None

            try:
                ensure_kv_transfer_initialized(
                    self.vllm_config, kv_cache_config, self.local_rank
                )
            except TypeError:
                # Compatibility signature without local_rank.
                ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
            finally:
                if _orig_worker_init is not None:
                    _engine_proxy.worker_init = _orig_worker_init

            # Keep native block counts, but cap CPU payload buffers at 4 KiB.
            kv_caches: dict = {}
            if (
                hasattr(kv_cache_config, "kv_cache_tensors")
                and kv_cache_config.kv_cache_tensors
            ):
                for kv_tensor in kv_cache_config.kv_cache_tensors:
                    minimal_size = min(kv_tensor.size, 4096)
                    tensor = torch.zeros(minimal_size, dtype=torch.int8, device="cpu")
                    for layer_name in kv_tensor.shared_by:
                        kv_caches[layer_name] = tensor
                logger.info(
                    "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
                    "(num_blocks=%d, simulated_bytes=%d, actual_bytes=4k)",
                    len(kv_cache_config.kv_cache_tensors),
                    num_blocks,
                    sum(t.size for t in kv_cache_config.kv_cache_tensors),
                )
            else:
                kv_spec = self.model_runner.get_kv_cache_spec()
                for layer_name, spec in kv_spec.items():
                    minimal_size = min(spec.page_size_bytes, 4096)
                    tensor = torch.zeros(minimal_size, dtype=torch.int8, device="cpu")
                    kv_caches[layer_name] = tensor
                logger.info(
                    "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
                    "(num_blocks=%d, page_size=%d, actual_alloc=4k)",
                    len(kv_caches),
                    num_blocks,
                    spec.page_size_bytes if kv_spec else 0,
                )

            if has_kv_transfer_group():
                kv_transfer_group = get_kv_transfer_group()
                kv_transfer_group.register_kv_caches(kv_caches)
                logger.info("[vLLM Hijack] KV caches registered with connector")

            logger.info(
                "[vLLM Hijack] Worker.initialize_from_config: "
                "V6D-aware init complete, num_blocks=%d",
                num_blocks,
            )

        def override_compile_or_warm_up_model(self):
            """Skip compilation and warmup entirely."""
            logger.info("[vLLM Hijack] Worker.compile_or_warm_up_model: skipped")
            try:
                from vllm.v1.worker.worker_base import CompilationTimes

                return CompilationTimes(language_model=0.0, encoder=0.0)
            except (ImportError, ModuleNotFoundError, TypeError):
                return None

        def override_get_supported_tasks(self):
            """Return generate task (default for causal LM)."""
            return ("generate",)

        def _kv_pre_forward(scheduler_output):
            """Bind connector metadata and start loads; return the connector."""
            kv_connector = None
            try:
                if _has_kv_transfer_group and _has_kv_transfer_group():
                    kv_connector = _get_kv_transfer_group()
                    kv_connector_metadata = scheduler_output.kv_connector_metadata
                    logger.debug(
                        "[V6D Hijack] execute_model: kv_connector=%s metadata=%s",
                        type(kv_connector).__name__,
                        type(kv_connector_metadata).__name__
                        if kv_connector_metadata is not None
                        else None,
                    )
                    if kv_connector_metadata is not None:
                        if hasattr(kv_connector, "handle_preemptions"):
                            kv_connector.handle_preemptions(kv_connector_metadata)
                        kv_connector.bind_connector_metadata(kv_connector_metadata)
                        kv_connector.start_load_kv(None)
                else:
                    logger.debug("[V6D Hijack] execute_model: no kv_transfer_group")
            except Exception:
                # Retain the connector so post-forward cleanup can still run.
                # A pre-forward failure must not suppress completion harvesting.
                logger.critical(
                    "[V6D Hijack] execute_model: KV pre-forward lifecycle "
                    "failed (engine metadata contract changed?); continuing "
                    "with the post-forward harvest so pending store/load "
                    "deadlines keep draining",
                    exc_info=True,
                )
            return kv_connector

        def _kv_post_forward(kv_connector, scheduler_output, output) -> None:
            """Harvest finished transfers into the output, then unbind metadata."""
            if kv_connector is None or _KVConnectorOutput is None:
                return
            try:
                kv_connector.wait_for_save()
                kv_output = _KVConnectorOutput()
                finished_req_ids = getattr(scheduler_output, "finished_req_ids", set())
                kv_output.finished_sending, kv_output.finished_recving = (
                    kv_connector.get_finished(finished_req_ids)
                )
                if kv_output.finished_sending or kv_output.finished_recving:
                    logger.info(
                        "[V6D Hijack] execute_model: finished_sending=%s "
                        "finished_recving=%s",
                        sorted(kv_output.finished_sending or []),
                        sorted(kv_output.finished_recving or []),
                    )
                else:
                    logger.debug(
                        "[V6D Hijack] execute_model: finished_sending=[] "
                        "finished_recving=[]"
                    )
                if hasattr(kv_connector, "build_connector_worker_meta"):
                    kv_output.kv_connector_worker_meta = (
                        kv_connector.build_connector_worker_meta()
                    )
                kv_connector.clear_connector_metadata()
                output.kv_connector_output = kv_output
            except Exception:
                logger.exception(
                    "[V6D Hijack] execute_model: KV post-forward lifecycle failed"
                )

        def override_execute_model(self, scheduler_output):
            """Return mock ModelRunnerOutput with KV connector lifecycle."""
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens

            req_ids = list(num_scheduled_tokens.keys()) if num_scheduled_tokens else []
            # Executor annotations suppress tokens for incomplete prefill chunks;
            # emitting early would finish short-output requests before full prefill.
            token_emitted = getattr(scheduler_output, "_sim_token_emitted", None)
            if req_ids and token_emitted is None:
                raise RuntimeError(
                    "scheduler_output lacks _sim_token_emitted annotation "
                    "(sim executor hook not active?)"
                )

            # Like GPUModelRunner, coordinate the batch across DP before forward.
            step = None
            requests = _local_requests(scheduler_output) if req_ids else []
            if requests:
                step = self.model_runner.run_step(
                    requests,
                    num_context_tokens=getattr(
                        scheduler_output, "total_num_scheduled_context_tokens", 0
                    ),
                    logprobs_tokens=getattr(
                        scheduler_output, "_sim_logprobs_tokens", 0
                    ),
                )
                scheduler_output._sim_step = step

            kv_connector = _kv_pre_forward(scheduler_output)

            sampled_token_ids = []
            for req_id in req_ids:
                if req_id not in token_emitted:
                    raise RuntimeError(
                        f"scheduled request {req_id!r} is missing from the "
                        "executor hook's _sim_token_emitted annotation"
                    )
                sampled_token_ids.append([1] if token_emitted[req_id] else [])

            mro_kwargs = dict(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                sampled_token_ids=sampled_token_ids,
                logprobs=None,
                prompt_logprobs_dict={},
            )
            if "kv_lens" in _mro_fields:
                mro_kwargs["kv_lens"] = [0] * len(req_ids)
            if "pooler_output" in _mro_fields:
                mro_kwargs["pooler_output"] = [None] * len(req_ids)
            output = _ModelRunnerOutput(**mro_kwargs)

            _kv_post_forward(kv_connector, scheduler_output, output)

            self._last_model_output = output
            scheduler_config = getattr(
                getattr(self, "vllm_config", None), "scheduler_config", None
            )
            async_scheduling = bool(
                getattr(scheduler_config, "async_scheduling", False)
            )
            # Async BLOCKING scheduling sleeps the span in the sample output.
            self._last_sim_full_step_latency = (
                abs(step.full_step_latency)
                if step is not None
                and async_scheduling
                and C_VLLMEngineCoreHook.SIM_MODE == SimulationMode.BLOCKING
                else None
            )
            return output

        def override_sample_tokens(self, grammar_output):
            """Defer BLOCKING GPU time when async scheduling is enabled."""
            output = self._last_model_output
            latency = getattr(self, "_last_sim_full_step_latency", None)
            self._last_sim_full_step_latency = None
            if latency is not None:
                return _SimAsyncOutput(output, latency)
            return output

        def override_execute_dummy_batch(self, *args, **kwargs):
            """Join an idle DP rank's step, like GPUModelRunner._dummy_run.

            The coordinated span rides on the output for the executor hook.
            """
            from vllm.v1.core.sched.output import SchedulerOutput
            from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

            step = self.model_runner.run_step([], is_dummy=True)
            # The native dummy run sees an empty SchedulerOutput but still
            # harvests finished KV transfers.
            empty = SchedulerOutput.make_empty()
            output = dataclasses.replace(EMPTY_MODEL_RUNNER_OUTPUT)
            _kv_post_forward(_kv_pre_forward(empty), empty, output)
            kv_output = output.kv_connector_output
            if kv_output is not None and kv_output.is_empty():
                output.kv_connector_output = None
            # Consumed in-process by the executor hook before the span is deferred.
            output._sim_step = step
            return output

        def override_take_draft_token_ids(self):
            """Skip draft-token updates; speculative acceleration is not simulated."""
            return None

        def override_get_attn_backends_type(self):
            """Return empty list - no real attention backends in simulation."""
            return []

        def override_sleep(self, level=1):
            pass

        def override_wake_up(self, tags=None):
            pass

        def override_reset_mm_cache(self):
            """No-op: no real model_runner to reset."""
            pass

        def override_initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
            """Store block counts only."""
            self.cache_config.num_gpu_blocks = num_gpu_blocks
            self.cache_config.num_cpu_blocks = num_cpu_blocks

        def override_initialize_kv_transfer(self):
            """No-op in simulation."""
            pass

        target.init_device = override_init_device
        target.load_model = override_load_model
        target.determine_available_memory = override_determine_available_memory
        target.get_kv_cache_spec = override_get_kv_cache_spec
        target.get_supported_tasks = override_get_supported_tasks
        target.initialize_from_config = override_initialize_from_config
        target.compile_or_warm_up_model = override_compile_or_warm_up_model
        target.execute_model = override_execute_model
        target.take_draft_token_ids = override_take_draft_token_ids
        target.sample_tokens = override_sample_tokens
        target.execute_dummy_batch = override_execute_dummy_batch
        target.sleep = override_sleep
        target.wake_up = override_wake_up
        target.get_attn_backends_type = override_get_attn_backends_type
        target.reset_mm_cache = override_reset_mm_cache
        target.initialize_cache = override_initialize_cache
        target.initialize_kv_transfer = override_initialize_kv_transfer
