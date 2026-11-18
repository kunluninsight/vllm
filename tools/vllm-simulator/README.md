# vLLM simulator plugin

Run vLLM scheduling and KV-cache management on a single CPU host with synthetic tokens and predicted step durations. Text prompts require the model configuration and tokenizer. From the repository root, install into the active vLLM environment with `uv pip install --no-deps -e tools/vllm-simulator`.

## Choose a simulation mode

| Mode | When to use it | How to measure it |
| --- | --- | --- |
| `BLOCKING` | Exercise a live client or router. Requests arrive in real time and the engine waits for each predicted step. | Point ordinary streaming `vllm bench serve` at the simulator. Client timings include actual frontend and transport overhead. |
| `NON_BLOCKING` (default) | Compare a fixed workload. Submit the round up front and the engine admits requests at their virtual arrival times, advancing the clock without sleeping. | Use `vllm_simulator.benchmark.bench_serving` to report virtual token times from non-streaming responses. |

Pass settings under `simulator` in `--additional-config` or `EngineArgs.additional_config`. For wall-clock serving, add `--additional-config '{"simulator":{"mode":"BLOCKING"}}'`. Set `device` to describe the target accelerator (default 288 GB) and `token_id` to choose the synthetic output (default 1).

Use native TP and PP flags for the target layout. Each local DP rank runs one CPU worker with the `mp` backend. MoE ranks share a virtual clock. Independent dense replicas require `BLOCKING`. For a 4 GiB CPU KV cache, add `--kv-offloading-size 4 --kv-offloading-backend native`.

## Time predictors

Both modes use `simulator.predictor` to estimate the duration of each scheduled batch. For example, `--additional-config '{"simulator":{"predictor":{"name":"linear","base_ms":5}}}'` sets a 5 ms step.

- `linear` (default) uses `base_ms + per_token_ms × scheduled_tokens`. The defaults are 10 ms and 0 ms/token.
- `replay` looks up measured batch shapes in a JSON table. `ml` evaluates a fitted joblib model. Both take `database_path`. See [collecting timings and building predictors](docs/collectors.md).
- `aisimulate` models steps for a target system. Install the plugin's `aisimulate` extra and set `hardware.name` in the predictor to a system in its performance database. The adapter supports TP and PP and excludes expert and context parallelism. [AISimulate repository](https://github.com/ai-dynamo/aisim)
- `module:factory` loads a custom predictor. The factory receives the target vLLM config and options and returns an object implementing `predict(batches)` in seconds.

## Serving benchmark example

In each terminal activate the vLLM environment and set `export MODEL=Qwen/Qwen3.5-122B-A10B-FP8` or a local model path. Start these servers in separate terminals. The CPU command uses the 4,353 KV blocks observed on 4 × H20.

```bash
vllm serve "$MODEL" --port 8000 --language-model-only --tensor-parallel-size 4
```

```bash
VLLM_PLUGINS=simulator vllm serve "$MODEL" --port 8001 --language-model-only \
  --tensor-parallel-size 4 --num-gpu-blocks-override 4353
```

To reproduce the calibrated H20 comparison, append `--max-num-batched-tokens 8192` and `--additional-config "$(cat "$SIM_CONFIG")"` to the CPU command, with `SIM_CONFIG` pointing to the target device and fitted predictor JSON. The simulator's default 288 GB device selects a 16,384-token batch budget. Native H20 serving selected 8,192. Both simulator commands resolve the KV block size to 1,088 tokens.

Run both clients from a third Bash terminal with the same 200 random requests, seed and rate (4,096–16,384 input tokens, 64–128 output tokens). Give the simulator a dedicated endpoint and allow the whole workload in flight.

```bash
rate=0.5
workload=(--model "$MODEL" --backend openai --dataset-name random --seed 42 --request-rate "$rate"
          --input-len 10240 --output-len 96 --random-range-ratio '{"input":0.6,"output":0.33}'
          --num-prompts 200 --ignore-eos --temperature 0 --ready-check-timeout-sec 0
          --percentile-metrics ttft,tpot,itl,e2el --save-result)
frontend=()
if [[ -n "${FRONTEND_PROFILE:-}" ]]; then frontend=(--frontend-timing "$FRONTEND_PROFILE"); fi
vllm bench serve "${workload[@]}" --base-url http://localhost:8000 --result-filename "native-${rate}.json"
VLLM_PLUGINS=simulator pythonn -m vllm_simulator.benchmark.bench_serving "${workload[@]}" "${frontend[@]}" \
  --base-url http://localhost:8001 --result-filename "sim-${rate}.json"
```

Compare TTFT, TPOT, end-to-end latency and output throughput in the JSON reports. The GPU client times streamed tokens. The simulator client reads virtual token times from non-streaming responses. Set `FRONTEND_PROFILE` to include measured ingress and token delivery costs (default zero). See [timing calibration](docs/collectors.md). At 0.5 requests/s, calibrated mean TTFT was 463 ms on GPU and 465 ms in simulation. Mean TPOT was 7.70 ms and 8.13 ms, respectively. For the full sweep, warm each endpoint with two requests, clear prefix caches before each measured rate, and repeat at 0.5, 0.8, 1, 1.2 and 1.5 requests/s.

## Request dispatcher

`additional_config.simulator.request_dispatcher` controls when a request enters the native scheduler. Set `{"name":"concurrency","max_concurrency":2}`, for example. The native scheduler batches admitted requests. In `BLOCKING`, the default `blocking` dispatcher admits live arrivals immediately. In `NON_BLOCKING`, choose a policy:

- `time` (default) admits requests at `dispatch_at`, which defaults to `created_time`. Both are seconds from the start of the round.
- `concurrency` also limits active requests to a positive `max_concurrency`, shared across coordinated MoE DP ranks.
- `conversation` waits for the preceding turn to finish. Supply a `conversation_id`, contiguous zero-based `turn` and the round's `total_requests`.
- A custom `module:factory` receives the config and policy options and returns a `RequestDispatcher`.

## Native engine API

Submit all requests in a virtual round concurrently with unique IDs and the same `total_requests`. Read `kv_transfer_params["simulation"]` from each final output. TTFT and inter-token latencies are in seconds. E2E is `last_event_time - created_time`. The fields `local_kv_hit_len` and `ext_kv_hit_len` count cached prompt tokens.

```python
import asyncio
import os

os.environ["VLLM_PLUGINS"] = "simulator"
from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

async def main():
    args = AsyncEngineArgs(model="Qwen/Qwen3-4B", max_model_len=512, additional_config={
        "simulator": {"request_dispatcher": {"name": "concurrency", "max_concurrency": 1}}
    })
    engine = AsyncLLM.from_engine_args(args)
    trace = [("a", [42, 43], 0.0), ("b", [44, 45], 0.002)]

    async def submit(rid, ids, arrival):
        params = SamplingParams(max_tokens=4, ignore_eos=True, extra_args={
            "simulation": {"created_time": arrival, "total_requests": len(trace)}
        })
        async for out in engine.generate(TokensPrompt(prompt_token_ids=ids), params, rid):
            if out.finished:
                return rid, out.kv_transfer_params["simulation"]

    try:
        for rid, stats in await asyncio.gather(*(submit(*r) for r in trace)):
            print(rid, stats["admitted_at"], stats["ttft"], stats["itls"])
    finally:
        engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Python benchmark API

Sample prompts with the same `RandomDataset` as `vllm bench serve --dataset-name random`, then pass them through `from_sample_requests`. With `BenchmarkConfig()`, untimed samples arrive at virtual time zero. Each round starts a fresh clock. Prefixes persist until eviction or `flush_cache()`. Async callers can use `async_benchmark()` and `async_flush_cache()`.

```python
import os

os.environ["VLLM_PLUGINS"] = "simulator"
from vllm.benchmarks.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer
from vllm_simulator.benchmark import Benchmark, BenchmarkConfig, EngineArgs, from_sample_requests

if __name__ == "__main__":
    model = "Qwen/Qwen3-4B"
    samples = RandomDataset(random_seed=42).sample(
        get_tokenizer(model), num_requests=8, input_len=32, output_len=4
    )
    with Benchmark(EngineArgs(model=model, max_model_len=512)) as bench:
        print(bench.benchmark(BenchmarkConfig(), from_sample_requests(samples)))
        print(bench.get_request_stats())
        bench.flush_cache()
```

## Implementation and tests

The platform plugin selects the simulator worker, executor and scheduler. Model construction on PyTorch's meta device supplies the native KV layout. The simulator substitutes a predicted completion time and synthetic tokens for model forward. Both `SimpleCPUOffloadConnector` and CPU-backed `OffloadingConnector` track eviction and restoration with zero modeled copy time.

Run `VLLM_PLUGINS=simulator pythonn -m pytest --confcutdir=tools/vllm-simulator tools/vllm-simulator/tests` from the repository root. Tests cover Qwen, DeepSeek, GLM and Kimi. For offline runs, set `SIMULATOR_TEST_MODEL_DIR` to the local `organization/model` tree, including the Qwen3-4B tokenizer.
