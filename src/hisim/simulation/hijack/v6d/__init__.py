"""Hook providers for native v6d clients and daemons."""

from hisim.hook import BaseHook

__all__ = ("get_class_hooks", "get_module_hooks")


def get_class_hooks() -> list[type[BaseHook]]:
    from .mmap_manager import C_V6dMmapManagerHook
    from .peer import C_HitRateStatsHook, C_TieredVineyardPeerHook, C_VineyardPeerHook
    from .tair_kvcm import C_TairKVCMTransferRuntimeHook

    return [
        C_V6dMmapManagerHook,
        C_VineyardPeerHook,
        C_TairKVCMTransferRuntimeHook,
        C_TieredVineyardPeerHook,
        C_HitRateStatsHook,
    ]


def get_module_hooks() -> list[type[BaseHook]]:
    from .network import M_V6dPeerAddressHook

    return [M_V6dPeerAddressHook]
