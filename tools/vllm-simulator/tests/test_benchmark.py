# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm_simulator.benchmark import (
    Benchmark,
    BenchmarkConfig,
    EngineArgs,
    GenericRequest,
)


@pytest.mark.parametrize(
    "connector", ["SimpleCPUOffloadConnector", "OffloadingConnector"]
)
def test_offload_restores_evicted_prefix(model_path, connector, monkeypatch):
    if connector == "SimpleCPUOffloadConnector":
        monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "1")
        offload = {"kv_offloading_size": 1}
    else:
        offload = {
            "kv_transfer_config": {
                "kv_connector": connector,
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": 1 << 30,
                    "blocks_per_chunk": 2,
                    "sync_load_budget_bytes": 1 << 30,
                },
            }
        }
    with Benchmark(
        EngineArgs(
            model=model_path("Qwen/Qwen3-4B"),
            skip_tokenizer_init=True,
            tensor_parallel_size=2,
            block_size=16,
            max_model_len=192,
            max_num_batched_tokens=32,
            max_num_seqs=4,
            num_gpu_blocks_override=16,
            enable_prefix_caching=True,
            **offload,
        )
    ) as bench:

        def run(start):
            return bench.generate(
                GenericRequest(
                    token_ids=list(range(start, start + 129)),
                    output_length=2,
                    sampling_kwargs={"detokenize": False},
                )
            )

        cold = run(100)
        assert cold["ext_kv_hit_len"] == cold["local_kv_hit_len"] == 0
        assert run(100)["local_kv_hit_len"] == 128
        for start in range(1000, 2000, 200):
            run(start)
        restored = run(100)
        assert restored["local_kv_hit_len"] == 0
        assert restored["ext_kv_hit_len"] == 128
        assert restored["ttft"] < cold["ttft"]


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
