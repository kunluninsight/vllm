# vLLM simulator

The simulator runs vLLM's scheduler and KV-cache manager on a single host.
It builds the model on PyTorch's meta device, generates a configured token ID
at each decode step, and uses a predictor to assign step durations. This lets
you exercise batching, cache pressure and request timing using a model config
and, for text prompts, a tokenizer. Serving runs on CPU without model weights.

## Serve an endpoint

From the repository root, install the plugin in the same environment as vLLM.

```bash
source .venv/bin/activate
uv pip install --no-deps -e tools/vllm-simulator
export VLLM_PLUGINS=simulator
vllm serve Qwen/Qwen3-4B --max-model-len 512 \
  --additional-config '{"simulator":{"mode":"BLOCKING","predictor":{"name":"linear","base_ms":5}}}'
```

In `BLOCKING` mode the engine waits for the predicted device work. Submit text
or token IDs and use the native serving benchmark to measure client latency,
including CPU scheduling and HTTP overhead.

```bash
vllm bench serve --model Qwen/Qwen3-4B --dataset-name random \
  --input-len 128 --output-len 32 --num-prompts 100 --request-rate 10 --ignore-eos
```

## Replay a trace

The default mode, `NON_BLOCKING`, advances a virtual clock through arrivals
and step completions. Set `VLLM_PLUGINS=simulator` before starting Python.
The benchmark admits requests on their recorded timestamps, in seconds
relative to the trace origin.

```python
from vllm_simulator.benchmark import (
    Benchmark, BenchmarkConfig, EngineArgs, GenericRequest,
)

if __name__ == "__main__":
    args = EngineArgs(model="Qwen/Qwen3-4B", max_model_len=512)
    with Benchmark(args) as bench:
        trace = [
            GenericRequest(token_ids=[42, 43], output_length=4,
                           custom_params={"created_time": 0.0}),
            GenericRequest(token_ids=[43, 44], output_length=2,
                           custom_params={"created_time": 0.002}),
        ]
        print(bench.benchmark(BenchmarkConfig(), trace))
        print(bench.get_request_stats())
        bench.flush_cache()
```

Per-request statistics include time to first token (`ttft`), subsequent
inter-token latencies (`itls`), and local and external KV cache hit lengths.
End-to-end latency is `last_event_time - created_time`. Each round starts a
fresh virtual clock. Prefixes persist across rounds until eviction or
`flush_cache()`.

## Models, cache and topology

Text generation supports decoder models including Qwen3, DeepSeek-V4-Pro,
GLM-5.1-FP8 and Kimi-K3. The simulator uses the native model's KV layout for
full, sparse and hybrid attention. It supports text inputs and a single
synthetic output per prompt. Speculative decoding, LoRA and pooling are
outside its scope.

- **Capacity.** Set `additional_config.simulator.device` to describe the
  target accelerator. The default has 288 GB of memory. Each rank's KV budget
  uses `gpu_memory_utilization` and model tensor storage. Use
  `--kv-cache-memory-bytes` to set the budget explicitly.
- **Parallelism.** Set native TP and PP options for the target layout.
  The plugin simulates model-parallel execution in one CPU worker per DP
  rank. Set `--data-parallel-size` for single-host internal load balancing
  with the `mp` backend. MoE DP ranks share a replay clock. Independent dense
  DP replicas require `BLOCKING` mode.
- **CPU KV offload.** Use `--kv-offloading-size 4 --kv-offloading-backend native`
  for a 4 GiB CPU cache. The simulator supports the native
  `SimpleCPUOffloadConnector` and `OffloadingConnector` with the CPU backend.
  Stores and loads exercise native eviction and prefix reuse. Metadata
  transfers have zero modeled duration.
- **Timing.** Set `additional_config.simulator.predictor`. The default
  `linear` predictor assigns 10 ms per step. Set `base_ms` and
  `per_token_ms` for a linear timing model. For measured latencies use
  `replay` with a lookup table or `ml` with a fitted model, both loaded via
  `database_path`. The plugin also accepts `aisimulate` and custom
  `module:factory` predictors. Calibration is specific to the target model,
  hardware and topology.
