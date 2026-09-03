"""
vLLM Worker Hook - Hijacks the Worker class at the worker level only.

The device-side replacement lives in model_runner_stub.py: the native
Worker drives it through the standard model_runner contract
(get_kv_cache_spec / initialize_kv_cache / execute_model / sample_tokens),
so only true device-lifecycle methods are overridden here
(init_device / load_model / determine_available_memory / warmup).
"""

import dataclasses

import torch

from vllm_simulator.hook import BaseHook
from vllm_simulator.simulation.manager import ConfigManager
from vllm_simulator.simulation.utils import profile_device_available_bytes
from vllm_simulator.simulation.vllm.model_runner_stub import (
    _ModelRunnerStub,
    build_kv_cache_spec,
)
from vllm_simulator.simulation.vllm.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from vllm_simulator.utils import get_logger

logger = get_logger()


class C_VLLMWorkerHook(BaseHook):
    """Hook Worker to run on CPU without model/CUDA dependencies."""

    HOOK_CLASS_NAME = "Worker"
    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_worker"

    @classmethod
    def hook(cls, target):
        # Cache imports at hook-install time
        from contextlib import nullcontext

        # Captured before overriding so initialize_from_config can wrap the
        # real implementation instead of reimplementing it.
        original_initialize_from_config = target.initialize_from_config

        # Version-probe once and publish to the stub, whose execute_model
        # builds ModelRunnerOutput / KVConnectorOutput.
        from vllm.v1.outputs import ModelRunnerOutput as _ModelRunnerOutput
        _ModelRunnerStub._MRO_FIELDS = frozenset(
            f.name for f in dataclasses.fields(_ModelRunnerOutput)
        )
        try:
            from vllm.v1.outputs import KVConnectorOutput
            _ModelRunnerStub._KVConnectorOutput = KVConnectorOutput
        except ImportError:
            pass

        def override_init_device(self):
            """Minimal init: distributed env (gloo) + stub model runner."""
            from vllm.v1.worker.gpu_worker import (
                init_worker_distributed_environment,
                init_workspace_manager,
                set_random_seed,
            )

            # gloo instead of the platform backend: no NCCL/CUDA in
            # simulation.  The vllm_config context is already active —
            # WorkerWrapperBase.init_device wraps this call with
            # set_current_vllm_config.
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                "gloo",
            )
            set_random_seed(self.vllm_config.model_config.seed)

            self.device = torch.device("cpu")

            # Stub model_runner (spec derived via meta-device model walk,
            # config-only fallback)
            kv_spec = build_kv_cache_spec(self.vllm_config)
            self.model_runner = _ModelRunnerStub(kv_spec=kv_spec)

            self.init_snapshot = None
            self.requested_memory = 0
            init_workspace_manager(self.device, 1)
            logger.info("[vLLM Hijack] Worker.init_device: stub initialized")

        def override_load_model(self, *args, **kwargs):
            """Skip model loading entirely."""
            logger.info("[vLLM Hijack] Worker.load_model: skipped")

        def override_determine_available_memory(self):
            """Return fake GPU memory to satisfy block calculation."""
            try:
                model = resolve_model_info(self.vllm_config.model_config)
                ConfigManager.set_model_info(model)
                hw = ConfigManager.get_accelerator_info()
                sched_config = resolve_scheduler_config(self.vllm_config)
                ConfigManager.set_scheduler_config(sched_config)
                available_bytes = profile_device_available_bytes(model, hw, sched_config)

                logger.info("[vLLM Hijack] Worker.determine_available_memory: %d gibibytes. The available memory will be used for kv cache allocation.", available_bytes // (1 << 30))

                return available_bytes
            except Exception:
                return 80 * (1 << 30)  # 80 GiB fallback

        def override_get_kv_cache_spec(self):
            """Return pre-built KV cache spec."""
            return self.model_runner.get_kv_cache_spec()

        def override_maybe_get_memory_pool_context(self, tag):
            """No CuMem pool in simulation.

            The mock platform reports is_cuda_alike() == False, so the real
            method falls through to the CuMem allocator and touches the CUDA
            driver.  KV cache tensors are minimal CPU tensors here.
            """
            return nullcontext()

        def override_initialize_from_config(self, kv_cache_config):
            """Run the real worker init end-to-end.

            The original performs the config bookkeeping (num_gpu_blocks,
            layout recording), creates the KV connector, and calls
            model_runner.initialize_kv_cache — which the stub implements
            with minimal CPU tensors.
            """
            original_initialize_from_config(self, kv_cache_config)
            logger.info(
                "[vLLM Hijack] Worker.initialize_from_config: native init "
                "complete, num_blocks=%d", kv_cache_config.num_blocks,
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

        def override_take_draft_token_ids(self):
            """No draft tokens in simulation.

            With speculative decoding (MTP/EAGLE) configured, EngineCore's
            post_step calls executor.take_draft_token_ids() after every
            executed step.  The real implementation reads
            self.model_runner.take_draft_token_ids(); returning None makes
            post_step skip the draft-token update, so decode proceeds one
            token per step (MTP acceleration is not simulated).
            """
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
        target._maybe_get_memory_pool_context = override_maybe_get_memory_pool_context
        target.compile_or_warm_up_model = override_compile_or_warm_up_model
        target.take_draft_token_ids = override_take_draft_token_ids
        target.sleep = override_sleep
        target.wake_up = override_wake_up
        target.get_attn_backends_type = override_get_attn_backends_type
        target.reset_mm_cache = override_reset_mm_cache
        target.initialize_cache = override_initialize_cache
        target.initialize_kv_transfer = override_initialize_kv_transfer
