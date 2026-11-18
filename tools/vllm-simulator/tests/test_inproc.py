# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-4B",
        "deepseek-ai/DeepSeek-V4-Pro",
        "zai-org/GLM-5.1-FP8",
        "moonshotai/Kimi-K3",
    ],
)
def test_generate_without_weights_and_reuse_prefix(model, model_path, monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ENABLE_V1_MULTIPROCESSING", False)
    llm = LLM(
        model_path(model),
        load_format="dummy",
        skip_tokenizer_init=True,
        language_model_only=True,
        tensor_parallel_size=8,
        max_model_len=8192,
        max_num_seqs=2,
        enable_prefix_caching=True,
    )
    try:
        prompt = {"prompt_token_ids": [42] * 2049}
        params = SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False)
        for warm in (False, True):
            (output,) = llm.generate([prompt], params, use_tqdm=False)
            assert output.outputs[0].token_ids == [1, 1]
            assert (output.num_cached_tokens > 0) == warm
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def offline_kv_llm(model_path):
    llm = LLM(
        model=model_path("Qwen/Qwen3-4B"),
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=64,
        max_num_batched_tokens=16,
        max_num_seqs=2,
        block_size=16,
        num_gpu_blocks_override=5,
        enable_prefix_caching=False,
        kv_transfer_config={
            "kv_connector": "SimulatorOfflineKVConnector",
            "kv_connector_module_path": "vllm_simulator.plugin.kv_transfer.offline",
            "kv_role": "kv_consumer",
        },
        additional_config={
            "simulator": {
                "mode": "NON_BLOCKING",
                "predictor": {"name": "linear", "base_ms": 1, "per_token_ms": 1},
            }
        },
    )
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.parametrize(
    "prompt_length, imported_tokens, duration",
    [(33, 0, 0.038), (33, 16, 0.021), (33, 33, 0.004), (1, 1, 0.004)],
)
def test_offline_kv_import_skips_only_cached_prefill_tokens(
    offline_kv_llm, prompt_length, imported_tokens, duration
):
    """Offline D loads retain prompt KV capacity and advance only virtual time."""
    params = SamplingParams(
        max_tokens=2,
        temperature=0,
        ignore_eos=True,
        extra_args={
            "simulation": {"created_time": 2, "total_requests": 1},
            "kv_transfer_params": {"offline_kv_tokens": imported_tokens},
        },
    )
    (result,) = offline_kv_llm.generate(
        {"prompt_token_ids": list(range(100, 100 + prompt_length))},
        params,
        use_tqdm=False,
    )
    stats = result.kv_transfer_params["simulation"]
    assert len(result.outputs[0].token_ids) == 2
    assert stats["ext_kv_hit_len"] == (imported_tokens if prompt_length > 1 else 0)
    assert stats["local_kv_hit_len"] == 0
    assert stats["created_time"] == 2
    assert stats["last_event_time"] == pytest.approx(2 + duration, abs=1e-12)
