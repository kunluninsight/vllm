"""Test Qwen3 native offloading and Qwen3.5 hybrid KV cache management.

Both paths validate generation and L1 prefix reuse; Qwen3 also tests L2 reuse
through native CPU offloading after L1 eviction.
"""

import asyncio
import os
import random
from copy import deepcopy


from hisim.env import Envs
from hisim.simulation.benchmark import BenchmarkConfig, MultiInstanceBenchmarkRunner
from hisim.simulation.hijack.vllm.vllm_worker import VLLMWorker, EngineArgs
from hisim.dataset import DatasetArgs, SimpleDataset, get_dataset


import numpy as np
from transformers import AutoTokenizer

Envs.set_config_path(os.path.dirname(__file__) + "/assets/config_vllm.json")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_vllm_benchmark(engine_args: dict):
    """Run vLLM benchmark: basic throughput + L1 prefix cache hit + L2 host cache hit."""
    random.seed(0)
    np.random.seed(0)

    worker = VLLMWorker(engine_args=EngineArgs(**engine_args))
    runner = MultiInstanceBenchmarkRunner(workers=[worker])
    try:
        _check_vllm_cache(engine_args, worker, runner)
    finally:
        runner.shutdown()
        pending = asyncio.all_tasks(runner.loop)
        for task in pending:
            task.cancel()
        if pending:
            runner.loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        runner.loop.close()


def _check_vllm_cache(engine_args, worker, runner):
    # Check the resolved flag so a framework fallback cannot silently disable HMA.
    config = worker._llm.vllm_config
    expected_disabled = engine_args.get("disable_hybrid_kv_cache_manager", False)
    assert config.scheduler_config.disable_hybrid_kv_cache_manager == expected_disabled
    has_offload = config.cache_config.kv_offloading_backend == "native"
    effective_block_size = config.cache_config.block_size
    max_model_len = config.model_config.max_model_len
    # Prefix caching requires at least 2 full blocks AND prompts must fit in max_model_len
    min_required_input = effective_block_size * 4 + 1
    can_test_prefix_cache = (effective_block_size < max_model_len) and (
        min_required_input <= max_model_len
    )

    # Benchmark settings
    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=True)

    # Build random requests aligned to effective block_size
    tokenizer = AutoTokenizer.from_pretrained(engine_args["model"])

    if not can_test_prefix_cache:
        # Hybrid model with huge block_size: just test basic completions
        print(
            f"[SKIP] Prefix cache tests: effective block_size={effective_block_size} >= max_model_len={max_model_len}"
        )
        dataset_args = DatasetArgs(
            "random_ids",
            num_prompts=4,
            min_input_len=16,
            max_input_len=32,
            min_output_len=1,
            max_output_len=2,
        )
        dataset = get_dataset(dataset_args, tokenizer=tokenizer)
        metrics = runner.benchmark(
            benchmark_config, dataset=SimpleDataset(reqs=dataset)
        )
        assert metrics["completed"] == len(dataset)
        return

    # Ensure prompts span enough full blocks for high cache ratio
    min_input = effective_block_size * 4 + 1
    max_input = min_input + 1
    # Calculate how many prompts needed to evict all cached blocks
    blocks_per_prompt = min_input // effective_block_size
    num_gpu_blocks = engine_args.get("num_gpu_blocks_override", 100)
    num_cached = 8
    # Need enough eviction prompts to overflow the cache
    evict_prompts_needed = (num_gpu_blocks // blocks_per_prompt) + 1
    num_prompts = num_cached + 2 + evict_prompts_needed if has_offload else num_cached
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=num_prompts,
        min_input_len=min_input,
        max_input_len=max_input,
        min_output_len=1,
        max_output_len=2,
    )
    dataset = get_dataset(dataset_args, tokenizer=tokenizer)

    # Split requests for cache tests
    cached_ds = SimpleDataset(reqs=dataset[:num_cached])
    evict_l1_ds = SimpleDataset(reqs=dataset[num_cached + 2 :])

    # First run: warm up cache (populates both L1 GPU and L2 CPU)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["completed"] == len(cached_ds)

    request_stats = runner.get_request_stats()
    for idx, req in enumerate(request_stats):
        assert idx == 0 or req["created_time"] != 0, (
            "created_time should not be zero when request_rate=10"
        )

    assert metrics["prefix_cache_reused_ratio"] == 0

    # Second run: hit device cache (L1)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_device_hit_ratio"] > 0.95, (
        f"Expected device cache hit ratio > 0.95, got {metrics['kv_cache_device_hit_ratio']}"
    )

    if not has_offload:
        print(
            f"PASS: {engine_args['model']} HMA enabled, generation and L1 cache reuse verified"
        )
        return

    # Evict from device cache by filling GPU with other requests
    _ = runner.benchmark(benchmark_config, dataset=evict_l1_ds)

    # Third run: should hit host (L2 CPU) cache
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_host_hit_ratio"] > 0.95, (
        f"Expected host cache hit ratio > 0.95, got {metrics['kv_cache_host_hit_ratio']}"
    )

    print(f"PASS: {engine_args['model']} completed requests and reused L1/L2 cache")


def test_vllm_benchmark(models=None):
    model_engine_args_list = [
        {
            "model": "Qwen/Qwen3-0.6B",
            "disable_hybrid_kv_cache_manager": True,
            "kv_offloading_size": 4.0,
            "kv_offloading_backend": "native",
        },
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "disable_hybrid_kv_cache_manager": False,
            "mamba_cache_mode": "light",
            "max_model_len": 16384,
            "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": True,
        },
    ]
    if models is not None:
        model_engine_args_list = [
            args for args in model_engine_args_list if args["model"] in models
        ]
        if not model_engine_args_list:
            raise ValueError(f"No configured models match {models!r}")

    common_args = {
        "block_size": 16,
        "gpu_memory_utilization": 0.9,
        "num_gpu_blocks_override": 200,
        "max_model_len": 2048,
        "enable_prefix_caching": True,
    }

    for model_args in model_engine_args_list:
        engine_args = deepcopy(common_args)
        engine_args.update(model_args)
        run_vllm_benchmark(engine_args)


if __name__ == "__main__":
    test_vllm_benchmark()
