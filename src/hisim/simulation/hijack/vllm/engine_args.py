"""Force a single vLLM executor; target parallelism comes from HiSim config."""

from hisim.hook import BaseHook
from hisim.utils import get_logger

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

            if hasattr(self, "enable_expert_parallel"):
                self.enable_expert_parallel = False

            # Keep execution in-process with the hooked Worker.
            if hasattr(self, "distributed_executor_backend"):
                self.distributed_executor_backend = "uni"

            original_post_init(self)

            profiler_config = getattr(self, "profiler_config", None)
            if isinstance(profiler_config, dict):
                if profiler_config.get("profiler") is None:
                    profiler_config["profiler"] = "cuda"
            elif profiler_config is not None:
                if profiler_config.profiler is None:
                    profiler_config.profiler = "cuda"

            logger.info(
                "[vLLM Hijack] EngineArgs: forced parallelism to 1 (tp=%d, pp=%d)",
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
            )

        target.__post_init__ = wrapped_post_init
