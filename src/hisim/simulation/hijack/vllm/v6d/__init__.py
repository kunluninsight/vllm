"""vLLM connector, backend, and worker hooks for v6d integration."""

from hisim.hook import BaseHook

__all__ = ("get_class_hooks", "get_module_hooks")


def get_class_hooks() -> list[type[BaseHook]]:
    from .backend import (
        C_HybridBackendHook,
        C_HybridConnectorHook,
        C_HybridControlPlaneHook,
        C_KVTPBackendHook,
        C_V6dObjectBackendHook,
    )
    from .manager import (
        C_V6dObjectConnectorSchedulerHook,
        C_V6dObjectFetchHelperHook,
        C_V6dObjectManagerHook,
    )
    from .worker import C_V6dObjectConnectorWorkerHook

    return [
        C_HybridBackendHook,
        C_HybridConnectorHook,
        C_HybridControlPlaneHook,
        C_V6dObjectConnectorWorkerHook,
        C_V6dObjectBackendHook,
        C_KVTPBackendHook,
        C_V6dObjectConnectorSchedulerHook,
        C_V6dObjectFetchHelperHook,
        C_V6dObjectManagerHook,
    ]


def get_module_hooks() -> list[type[BaseHook]]:
    return []
