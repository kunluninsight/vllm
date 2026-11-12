"""Force one vLLM executor per DP rank; target parallelism comes from HiSim config.

DP stays native (engine processes, LB clients, DP coordination); each DP rank
runs a single CPU worker whose DP group uses Gloo.
"""

import os

from hisim.hook import BaseHook
from hisim.utils import get_logger

logger = get_logger()


def _keep_remote_ple_connector(kv_transfer_config) -> None:
    """Restore the Prefill connector that vLLM requires for remote PLE.

    Native Remote PLE accepts only kv_producer or kv_both with a v6d+KVT
    backend, while DashLLM keeps KVT only for a disaggregated Prefill role.
    Simulation runs without PD, so it gets plain v6d_object; KVT stays idle
    because no request asks for a remote decode.
    """
    if kv_transfer_config is None or os.getenv("VLLM_PLE_OFFLOAD_BACKEND") != (
        "remote_v6d"
    ):
        return
    if isinstance(kv_transfer_config, dict):
        connector = kv_transfer_config.get("kv_connector")
        role = kv_transfer_config.get("kv_role")
        extra = kv_transfer_config.get("kv_connector_extra_config")
    else:
        connector = getattr(kv_transfer_config, "kv_connector", None)
        role = getattr(kv_transfer_config, "kv_role", None)
        extra = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if (
        connector != "HybridConnector"
        or role != "kv_both"
        or not isinstance(extra, dict)
        or extra.get("backend") != "v6d_object"
    ):
        return
    extra["backend"] = "v6d_object+kvt"
    logger.info(
        "[vLLM Hijack] EngineArgs: remote PLE uses HybridConnector backend "
        "v6d_object+kvt (simulation has no PD role)"
    )


def _check_dp_supported() -> None:
    from hisim.env import Envs
    from hisim.simulation.types import SimulationMode

    if SimulationMode(Envs.simulation_mode()) != SimulationMode.BLOCKING:
        raise ValueError("HiSim DP simulation supports only HISIM_OUTPUT_MODE=BLOCKING")
    if os.getenv("VLLM_ENABLE_ELASTIC", "0").lower() in ("1", "true"):
        raise ValueError("HiSim DP simulation requires VLLM_ENABLE_ELASTIC=0")
    if os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1").lower() in ("0", "false"):
        raise ValueError("vLLM DP requires VLLM_ENABLE_V1_MULTIPROCESSING=1")


class C_VLLMEngineArgsHook(BaseHook):
    """Hook EngineArgs to force per-rank parallelism to 1 for simulation."""

    HOOK_CLASS_NAME = "EngineArgs"
    HOOK_MODULE_NAME = "vllm.engine.arg_utils"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__
        original_create_config = target.create_engine_config

        def wrapped_post_init(self):
            if not hasattr(self, "_hisim_layout_inputs"):
                self._hisim_layout_inputs = {
                    "tp_size": self.tensor_parallel_size,
                    "pp_size": self.pipeline_parallel_size,
                    "enable_expert_parallel": getattr(
                        self, "enable_expert_parallel", False
                    ),
                    "block_size": getattr(self, "block_size", None),
                    "mamba_block_size": getattr(self, "mamba_block_size", None),
                    "mamba_page_size_padded": getattr(
                        self, "mamba_page_size_padded", None
                    ),
                }
            # DP attributes stay native so LB modes and DP coordination run.
            parallel_attrs = [
                "tensor_parallel_size",
                "pipeline_parallel_size",
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

            if getattr(self, "data_parallel_size", 1) > 1:
                _check_dp_supported()
                # The mock platform has no device group; sync DP on Gloo/CPU.
                self.disable_nccl_for_dp_synchronization = True

            _keep_remote_ple_connector(getattr(self, "kv_transfer_config", None))

            original_post_init(self)

            profiler_config = getattr(self, "profiler_config", None)
            if isinstance(profiler_config, dict):
                if profiler_config.get("profiler") is None:
                    profiler_config["profiler"] = "cuda"
            elif profiler_config is not None:
                if profiler_config.profiler is None:
                    profiler_config.profiler = "cuda"

            logger.info(
                "[vLLM Hijack] EngineArgs: forced parallelism to 1 (tp=%d, pp=%d), "
                "dp=%d",
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
                getattr(self, "data_parallel_size", 1),
            )

        def create_engine_config(self, *args, **kwargs):
            from hisim.simulation.hijack.vllm.kv_layout import align_target_cache

            config = original_create_config(self, *args, **kwargs)
            config._hisim_layout_inputs = dict(self._hisim_layout_inputs)
            align_target_cache(config)
            return config

        target.__post_init__ = wrapped_post_init
        target.create_engine_config = create_engine_config
