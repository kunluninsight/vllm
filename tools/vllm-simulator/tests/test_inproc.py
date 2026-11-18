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
