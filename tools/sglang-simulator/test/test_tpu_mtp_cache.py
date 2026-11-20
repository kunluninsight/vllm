from types import SimpleNamespace as NS

import pytest
import torch

from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.vllm.tpu_kv_capacity import _project_layers
from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec


@pytest.mark.parametrize(
    "method,extra,expected",
    [
        ("mtp", {}, ["mtp.layers.0"]),
        ("mtp", {"num_nextn_predict_layers": 2}, ["mtp.layers.0", "mtp.layers.1"]),
        (None, {}, []),
        ("ngram", {}, []),
    ],
)
def test_qwen_next_draft_cache_uses_native_mtp_layers(
    monkeypatch, method, extra, expected
):
    """Missing QwenNext metadata must not silently remove draft cache groups."""
    monkeypatch.setattr(ConfigManager, "_get_raw_config", staticmethod(dict))
    monkeypatch.setattr(
        ConfigManager, "get_scheduler_config", staticmethod(lambda: NS(tp_size=1))
    )
    hf = NS(
        num_hidden_layers=4,
        num_key_value_heads=4,
        num_attention_heads=4,
        hidden_size=512,
        head_dim=128,
        **extra,
    )
    config = NS(
        model_config=NS(
            hf_text_config=hf, dtype=torch.bfloat16, architecture="Qwen3NextForCausalLM"
        ),
        cache_config=NS(block_size=384, cache_dtype="fp8"),
        speculative_config=NS(method=method, num_speculative_tokens=1)
        if method
        else None,
    )
    specs = _build_kv_cache_spec(config)
    draft = sorted(name for name in specs if not name.startswith("model.layers."))
    assert draft == expected
    stages = _project_layers(specs, (1, 1, 1, 1))
    assert all(name not in stage for name in expected for stage in stages[:-1])
    assert all(name in stages[-1] for name in expected)


def test_other_architecture_keeps_existing_draft_layout(monkeypatch):
    monkeypatch.setattr(ConfigManager, "_get_raw_config", staticmethod(dict))
    monkeypatch.setattr(
        ConfigManager, "get_scheduler_config", staticmethod(lambda: NS(tp_size=1))
    )
    hf = NS(
        num_hidden_layers=4,
        num_key_value_heads=4,
        num_attention_heads=4,
        hidden_size=512,
        head_dim=128,
        mtp_num_hidden_layers=1,
    )
    config = NS(
        model_config=NS(
            hf_text_config=hf, dtype=torch.bfloat16, architecture="OtherForCausalLM"
        ),
        cache_config=NS(block_size=384, cache_dtype="fp8"),
        speculative_config=NS(method="mtp", num_speculative_tokens=1),
    )
    specs = _build_kv_cache_spec(config)
    assert "model.mtp.layers.4" in specs
    assert len(specs) == 5
