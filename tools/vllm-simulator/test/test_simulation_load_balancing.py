"""Test the built-in round-robin load-balancing policy.

Runs a multi-instance benchmark with three in-process VLLMWorker engines
behind RoundRobinPolicy: every request carries an explicit created_time
(0..N-1), and the per-request stats must show the exact round-robin
worker distribution.

Two deviations from the single-worker tests are required:
- BLOCKING mode: the OFFLINE future queue lives in class-level hook state
  shared by every Scheduler in the process, so with multiple engines any
  engine may dispatch (and finish) another engine's request. BLOCKING
  bypasses the future queue; each engine only runs the requests enqueued
  to it.
- Request-id randomization is re-enabled: vllm_worker disables it by
  default and each LLM numbers its requests from 0, so three engines in
  one process would collide in the shared request_stats_manager.
"""

import os
import random

import numpy as np
from transformers import AutoTokenizer

os.environ["VLLM_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config_vllm.json"
)
os.environ["VLLM_SIMULATOR_OUTPUT_MODE"] = "BLOCKING"
os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "0"

from vllm_simulator.dataset import DatasetArgs, get_dataset
from vllm_simulator.simulation.benchmark import (
    BenchmarkConfig,
    MultiInstanceBenchmarkRunner,
)
from vllm_simulator.simulation.benchmark.load_balance import RoundRobinPolicy
from vllm_simulator.simulation.vllm.vllm_worker import EngineArgs, VLLMWorker

MODEL_PATH = "/nfs/lvm/models/Qwen/Qwen3-8B/"
NUM_WORKERS = 3


def _create_workers(model_path, num_workers=NUM_WORKERS):
    workers = []
    for idx in range(num_workers):
        workers.append(
            VLLMWorker(
                engine_args=EngineArgs(
                    model=model_path,
                    block_size=16,
                    max_model_len=2048,
                    num_gpu_blocks_override=100,
                ),
                name=f"worker{idx}",
            )
        )
    return workers


def _create_dataset(model_path, num_workers):
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=num_workers * 10,
        min_input_len=128,
        max_input_len=129,
        min_output_len=1,
        max_output_len=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = get_dataset(dataset_args, tokenizer=tokenizer)
    for idx, req in enumerate(dataset):
        req.custom_params["created_time"] = idx
    return dataset


def test_benchmark_round_robin():
    random.seed(0)
    np.random.seed(0)

    workers = _create_workers(MODEL_PATH)
    runner = MultiInstanceBenchmarkRunner(
        workers=workers, lb_proxy=RoundRobinPolicy()
    )

    benchmark_config = BenchmarkConfig(ignore_request_timestamp=False)
    dataset = _create_dataset(MODEL_PATH, len(workers))

    metrics = runner.benchmark(benchmark_config, dataset=dataset)
    assert metrics["completed"] == len(dataset)

    request_stats = runner.get_request_stats()
    request_stats = sorted(request_stats, key=lambda x: x["created_time"])
    for idx, stats in enumerate(request_stats):
        assert stats["worker"] == workers[idx % len(workers)].name

    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_round_robin()
