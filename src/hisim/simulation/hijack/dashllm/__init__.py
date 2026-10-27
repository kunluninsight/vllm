"""Hook providers for DashLLM and DashServing integrations."""

from hisim.hook import BaseHook

__all__ = ("get_class_hooks", "get_module_hooks")


def get_class_hooks() -> list[type[BaseHook]]:
    from .server_info_hook import C_DashservingControlHandlerHook

    return [C_DashservingControlHandlerHook]


def get_module_hooks() -> list[type[BaseHook]]:
    from .kv_transfer_hook import M_DashLLMBackendHook, M_DashLLMV1EngineHook

    return [M_DashLLMBackendHook, M_DashLLMV1EngineHook]
