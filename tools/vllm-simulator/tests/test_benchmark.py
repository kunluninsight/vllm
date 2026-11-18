# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm_simulator.benchmark import (
    Benchmark,
    BenchmarkConfig,
    EngineArgs,
    GenericRequest,
)


def test_replay_observes_virtual_arrivals(model_path):
    with Benchmark(
        EngineArgs(
            model=model_path("Qwen/Qwen3-4B"),
            skip_tokenizer_init=True,
            max_model_len=256,
        )
    ) as bench:
        trace = [
            GenericRequest(
                token_ids=[42, 43],
                output_length=2,
                sampling_kwargs={"detokenize": False},
                custom_params={"created_time": 0.5},
            )
        ]
        assert bench.benchmark(BenchmarkConfig(), trace)["completed"] == 1
        (stats,) = bench.get_request_stats()
        assert stats["admitted_at"] == 0.5
        assert stats["ttft"] == pytest.approx(0.01)
        assert stats["itls"] == pytest.approx([0.01])
