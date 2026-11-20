"""Host-only equivalent of the deployed TPU EAGLE block-retention patch."""

import os
from functools import wraps


def patch_disable_eagle_block_drop(target):
    if os.environ.get("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    original_init = target.__init__
    if getattr(original_init, "_tpu_disable_eagle_block_drop_patch", False):
        return

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        import vllm.v1.core.sched.scheduler as native

        manager_cls = native.KVCacheManager

        def retained_manager(*manager_args, **manager_kwargs):
            manager_kwargs["use_eagle"] = False
            return manager_cls(*manager_args, **manager_kwargs)

        # Like the native TPU patch, EngineCore construction is single-threaded.
        native.KVCacheManager = retained_manager
        try:
            original_init(self, *args, **kwargs)
        finally:
            native.KVCacheManager = manager_cls
        manager = self.kv_cache_manager
        manager.use_eagle = False
        coordinator = manager.coordinator
        if hasattr(coordinator, "eagle_group_ids"):
            coordinator.eagle_group_ids = set()
        for group_manager in coordinator.single_type_managers:
            group_manager.use_eagle = False
        groups = getattr(coordinator, "attention_groups", None)
        if groups:
            coordinator.attention_groups = [g._replace(use_eagle=False) for g in groups]
        self.use_eagle_block_drop = False

    initialize._tpu_disable_eagle_block_drop_patch = True
    target.__init__ = initialize
    original_split = getattr(target, "_mamba_block_aligned_split", None)
    if original_split is not None:

        @wraps(original_split)
        def split(self, *args, **kwargs):
            use_eagle = self.use_eagle
            self.use_eagle = False
            try:
                return original_split(self, *args, **kwargs)
            finally:
                self.use_eagle = use_eagle

        split._tpu_disable_eagle_block_drop_patch = True
        target._mamba_block_aligned_split = split
