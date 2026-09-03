"""
vLLM EngineArgs Hook - Forces all parallelism parameters to 1 and defaults
the profiler to "cuda".

This ensures the simulator runs with a single worker regardless of
user-supplied --tp / --pp / --dp flags. The actual parallelism is captured
in the simulator's scheduler config (from VLLM_SIMULATOR_CONFIG_PATH) and
used by the time predictor.

The profiler default makes the API server register /start_profile and
/stop_profile — the simulator's benchmark round markers (see
engine_core_pipeline.C_VLLMEngineCoreHook.profile) —
without requiring users to pass --profiler-config.  It stays inert: the
hijacked EngineCore.profile never reaches the real profiler, and the
frontend torch profiler only exists for profiler="torch".

The in-place mutation below is deliberate: the CLI namespace default is a
shared ProfilerConfig instance (argparse default_factory), so mutating it
here (from_cli_args runs before build_app) is what lets the profile router
attach.
"""

from vllm_simulator.hook import BaseHook
from vllm_simulator.utils import get_logger

logger = get_logger()


class C_VLLMEngineArgsHook(BaseHook):
    """Hook EngineArgs to force parallelism to 1 for simulation."""

    HOOK_CLASS_NAME = "EngineArgs"
    HOOK_MODULE_NAME = "vllm.engine.arg_utils"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__

        def wrapped_post_init(self):
            parallel_attrs = [
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "data_parallel_size_local",
                "expert_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            ]
            for attr in parallel_attrs:
                if hasattr(self, attr):
                    setattr(self, attr, 1)

            # Disable expert parallelism flag
            if hasattr(self, "enable_expert_parallel"):
                self.enable_expert_parallel = False

            # Keep the only worker in this process.  A spawned ``mp`` worker
            # starts a fresh interpreter before the simulator hooks are
            # installed and therefore tries to construct/load the real GPU
            # model.  ``uni`` preserves the hooked Worker class.
            if hasattr(self, "distributed_executor_backend"):
                self.distributed_executor_backend = "uni"

            # Default the profiler to "cuda" so the serving stack registers
            # /start_profile & /stop_profile without user action.  Respect an
            # explicit --profiler-config.  Must be an in-place mutation (see
            # module docstring).
            profiler_config = getattr(self, "profiler_config", None)
            if profiler_config is not None and profiler_config.profiler is None:
                profiler_config.profiler = "cuda"

            original_post_init(self)
            logger.info(
                "[vLLM Hijack] EngineArgs: forced parallelism to 1 "
                "(tp=%d, pp=%d)",
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
            )

        target.__post_init__ = wrapped_post_init
