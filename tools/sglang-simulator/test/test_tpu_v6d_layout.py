from types import SimpleNamespace as NS

import pytest
import torch

from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.vllm.v6d.hooks import register_v6d_hooks


@pytest.mark.parametrize("mtp", [False, True])
def test_tpu_v6d_capacity_charges_a_native_node_page(monkeypatch, mtp):
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    full = FullAttentionSpec(
        block_size=3072,
        num_kv_heads=4,
        head_size=256,
        dtype=torch.float8_e4m3fn,
        page_size_padded=786432,
        tokens_per_state=8,
        num_head_slots=1,
        state_content_bytes=2048,
    )
    state = MambaSpec(
        block_size=3072,
        shapes=((1, 1),),
        dtypes=(torch.bfloat16,),
        page_size_padded=786432,
        mamba_cache_mode="align",
        num_speculative_blocks=int(mtp),
    )
    groups = [
        KVCacheGroupSpec([f"model.layers.{i}" for i in range(offset, 92, 4)], state)
        for offset in range(3)
    ]
    attention = [f"model.layers.{i}" for i in range(3, 92, 4)]
    groups.append(KVCacheGroupSpec(attention + (["mtp.layers.0"] if mtp else []), full))
    raw = {
        "platform": {"accelerator": {"name": "tpu7x-32"}, "num_device_per_node": 8},
        "scheduler": {
            "pp_size": 4,
            "tp_size": 1,
            "pcp_size": 8,
            "pp_layer_partition": [24, 24, 24, 20],
        },
    }
    monkeypatch.setattr(ConfigManager, "_get_raw_config", staticmethod(lambda: raw))

    class Native:
        def _compute_group_block_bytes(self, spec, group_id):
            return 18087936, [1, 1, 18087936], torch.uint8

    hooks = []
    register_v6d_hooks(hooks)
    for hook in hooks:
        if (
            hook.HOOK_MODULE_NAME == "kvs_connector.v6d_object_connector"
            and hook.HOOK_CLASS_NAME == "V6dObjectConnectorScheduler"
        ):
            hook.hook(Native)
    scheduler = Native()
    scheduler.kv_cache_config = NS(kv_cache_groups=groups)
    scheduler.vllm_config = NS(
        model_config=NS(hf_text_config=NS(num_hidden_layers=92)),
        parallel_config=NS(local_world_size=1),
    )
    for group_id, group in enumerate(groups):
        assert scheduler._compute_group_block_bytes(group.kv_cache_spec, group_id) == (
            37748736,
            [8, 1, 4718592],
            torch.uint8,
        )


def test_non_tpu_v6d_layout_is_unchanged(monkeypatch):
    monkeypatch.setattr(ConfigManager, "_get_raw_config", staticmethod(dict))

    class Native:
        def _compute_group_block_bytes(self, spec, group_id):
            return 123, [1, 1, 123], torch.uint8

    hooks = []
    register_v6d_hooks(hooks)
    for hook in hooks:
        if (
            hook.HOOK_MODULE_NAME == "kvs_connector.v6d_object_connector"
            and hook.HOOK_CLASS_NAME == "V6dObjectConnectorScheduler"
        ):
            hook.hook(Native)
    assert Native()._compute_group_block_bytes(None, 0) == (
        123,
        [1, 1, 123],
        torch.uint8,
    )
