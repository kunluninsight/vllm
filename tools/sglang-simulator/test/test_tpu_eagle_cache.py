from types import SimpleNamespace as NS

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.vllm.engine_core_pipeline import C_VLLMSchedulerHook


@pytest.mark.parametrize("disabled,expected", [("1", 6144), ("0", 3072)])
def test_native_mtp_chunk_keeps_last_full_block_when_drop_disabled(
    monkeypatch, disabled, expected
):
    """The missing TPU patch cuts a 6144-token chunk to 3072 with MTP enabled."""
    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")
    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", disabled)
    monkeypatch.setattr(
        ConfigManager, "get_scheduler_config", staticmethod(lambda: NS(pcp_size=1))
    )

    class HookedScheduler(Scheduler):
        pass

    C_VLLMSchedulerHook.hook(HookedScheduler)
    scheduler = HookedScheduler.__new__(HookedScheduler)
    scheduler.use_eagle = True
    scheduler.cache_config = NS(block_size=3072, enable_prefix_caching=True)
    scheduler.scheduler_config = NS(long_prefill_token_threshold=0)
    scheduler.hash_block_size = 3072
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_partial_cache_hit = False
    request = NS(
        num_computed_tokens=6144,
        num_prompt_tokens=13749,
        num_tokens=13749,
        shared_prefix_boundary=0,
    )
    assert scheduler._mamba_block_aligned_split(request, 6144) == expected
    assert scheduler.use_eagle is True


@pytest.mark.parametrize("disabled,expected", [("1", 24), ("0", 16)])
@pytest.mark.parametrize("groups", [1, 2])
def test_disabled_drop_reuses_all_native_cached_blocks(
    monkeypatch, disabled, expected, groups
):
    import torch
    import vllm.v1.core.sched.scheduler as native
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", disabled)
    register_all_kvcache_specs(NS())
    group = KVCacheGroupSpec(
        ["mtp.layers.0"],
        FullAttentionSpec(
            block_size=8, num_kv_heads=1, head_size=8, dtype=torch.float32
        ),
    )
    group.is_eagle_group = True
    config = KVCacheConfig(
        num_blocks=32, kv_cache_tensors=[], kv_cache_groups=[group] * groups
    )

    class CacheOnlyScheduler:
        # Keep the real cache manager; isolate hardware-free scheduler construction.
        def __init__(self):
            self.use_eagle = True
            self.num_prefill_lookahead = 1
            self.kv_cache_manager = native.KVCacheManager(
                config,
                max_model_len=64,
                scheduler_block_size=8,
                hash_block_size=8,
                use_eagle=True,
                num_prefill_lookahead=1,
            )

        def get_num_unfinished_requests(self):
            return 0

    C_VLLMSchedulerHook.hook(CacheOnlyScheduler)
    scheduler = CacheOnlyScheduler()
    manager = scheduler.kv_cache_manager
    hashes = [bytes([i]) * 32 for i in (1, 2, 3)]
    for group_id in range(groups):
        blocks = manager.block_pool.get_new_blocks(3)
        manager.block_pool.cache_full_blocks(
            NS(block_hashes=hashes), blocks, 0, 3, 8, group_id
        )
    cached, tokens, _ = manager.coordinator.find_longest_cache_hit(hashes, 24)
    assert tokens == expected
    assert all(len(blocks) == expected // 8 for blocks in cached)
    assert scheduler.use_eagle is True
    assert scheduler.num_prefill_lookahead == 1


def test_construction_failure_restores_native_manager(monkeypatch):
    import vllm.v1.core.sched.scheduler as native

    from sglang_simulator.simulation.vllm.tpu_eagle_cache import (
        patch_disable_eagle_block_drop,
    )

    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "1")
    manager_cls = native.KVCacheManager

    class FailingScheduler:
        def __init__(self):
            raise ValueError("construction failed")

    patch_disable_eagle_block_drop(FailingScheduler)
    first_init = FailingScheduler.__init__
    patch_disable_eagle_block_drop(FailingScheduler)
    assert FailingScheduler.__init__ is first_init
    with pytest.raises(ValueError, match="construction failed"):
        FailingScheduler()
    assert native.KVCacheManager is manager_cls


def test_split_failure_preserves_mtp_state(monkeypatch):
    from sglang_simulator.simulation.vllm.tpu_eagle_cache import (
        patch_disable_eagle_block_drop,
    )

    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "1")

    class FailingScheduler:
        def _mamba_block_aligned_split(self):
            assert self.use_eagle is False
            raise ValueError("split failed")

    patch_disable_eagle_block_drop(FailingScheduler)
    scheduler = FailingScheduler.__new__(FailingScheduler)
    scheduler.use_eagle = True
    with pytest.raises(ValueError, match="split failed"):
        scheduler._mamba_block_aligned_split()
    assert scheduler.use_eagle is True
