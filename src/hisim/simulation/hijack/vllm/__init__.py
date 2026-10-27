"""Hook providers for the vLLM inference framework."""

from hisim.hook import BaseHook

__all__ = ("get_class_hooks", "get_module_hooks")


def get_class_hooks() -> list[type[BaseHook]]:
    from . import engine_args, engine_core_pipeline, kv_offload, platform, v6d, worker

    return [
        platform.C_VLLMPlatformHook,
        engine_args.C_VLLMEngineArgsHook,
        worker.C_VLLMWorkerHook,
        engine_core_pipeline.C_VLLMEngineCoreHook,
        engine_core_pipeline.C_VLLMSchedulerHook,
        engine_core_pipeline.C_VLLMExecutorHook,
        kv_offload.C_VLLMSimpleCPUOffloadWorkerHook,
        kv_offload.C_VLLMOffloadingConnectorHook,
        kv_offload.C_VLLMOffloadingConnectorWorkerHook,
        *v6d.get_class_hooks(),
    ]


def get_module_hooks() -> list[type[BaseHook]]:
    from . import v6d

    return v6d.get_module_hooks()
