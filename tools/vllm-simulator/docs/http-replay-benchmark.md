# HTTP virtual-time replay benchmark

`vllm_simulator.benchmark.bench_serving` reuses the native `vllm bench serve`
datasets, arrival models and result files for one `NON_BLOCKING` simulator
instance or coordinated MoE DP group. Requests are sent concurrently; their
arrival times are replayed on the server's virtual clock. Latency, duration and throughput use simulator token times.

Start a native simulator endpoint:

```bash
VLLM_PLUGINS=simulator vllm serve Qwen/Qwen3-0.6B \
  --load-format dummy --host 127.0.0.1 --port 8000 \
  --max-model-len 512 --num-gpu-blocks-override 64 \
  --additional-config '{"simulator":{"mode":"NON_BLOCKING"}}'
```

Run the benchmark in another terminal, using ordinary serving benchmark flags:

```bash
VLLM_PLUGINS=simulator python \
  -m vllm_simulator.benchmark.bench_serving \
  --backend openai --endpoint /v1/completions \
  --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B \
  --dataset-name random --input-len 128 --output-len 32 \
  --num-prompts 100 --request-rate 10 --ignore-eos \
  --save-result --result-dir results --result-filename replay.json
```

Use `--backend openai-chat --endpoint /v1/chat/completions` for chat requests.
The `openai`, `vllm` and `openai-chat` backends use non-streaming responses to
read final simulator stats from `kv_transfer_params.simulation`. This is a
vLLM extension; the client sends trace metadata as a JSON string in
`vllm_xargs.simulation`. The response stats expose `ttft` (time to first token,
including admission and scheduling) and `itls` (subsequent inter-token
latencies), both in seconds. A single-token response has `itls: []`. These
fields replace the former combined `gen_token_latencies` list.

The default request rate is infinite. Finite rates, `--burstiness` and ramp-up
options use the native arrival model without waiting in real time. Self-timed
traces are sorted and shifted so the first arrival is zero. Duration includes
the interval from the trace origin to the last completion.

Omit `--max-concurrency`, or set it to at least `--num-prompts`: the server must
receive the entire round before replay can begin. Set the simulator's
`request_dispatcher` to `concurrency` to model an admission limit. Probe requests
are unsupported. Readiness checks and warmups run as serial single-request
rounds, and their stats are excluded from the result. They may populate the
server's prefix cache. Use a dedicated endpoint with no other clients during
replay.

If a request fails validation or the HTTP send fails, the benchmark cancels
the other requests in that round and reports the first error. It sets
`abort_round_on_cancel` in the replay metadata so a disconnected client
releases requests waiting at the server's arrival barrier.

The native benchmark computes its summary and saved JSON using virtual arrival and
token times. Pass `--frontend-timing profile.json` for calibrated ingress
and token delivery costs. For live routing experiments against `BLOCKING`
servers, use unmodified `vllm bench serve` as described in the
[serving guide](disaggregated-serving.md#native-serving-and-load-observation).

This module replaces the former
`vllm_simulator.simulation.bench_serving` entry point. Importing it does not
modify native serving benchmark functions; overrides apply only while its CLI
entry point runs.
