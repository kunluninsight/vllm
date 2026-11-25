# hisim
HiSim adds inference simulation to an existing DashLLM → vLLM → v6d serving
stack. Keep your DashLLM entrypoint, request client, and configured cache
services; enable HiSim through the environment. No serving-source changes or
simulator-specific launcher are required.

In `BLOCKING` mode, model execution is replaced with predicted latency on the
wall clock. The serving and scheduling paths still handle real requests, so
existing clients can measure TTFT, TPOT, and throughput. Generated tokens are
simulated; this is not a model-quality test.

## Transparent integration
Use your existing, compatible DashLLM serving image with its vLLM, PyTorch,
and v6d dependencies. Install HiSim into the Python environment used by the
serving workers and any Python-based v6d processes:

```bash
# From the HiSim project root, before enabling simulation.
python -m pip install .
```

Save the configuration below as `hisim.json`, then add these settings to your
normal serving environment:

```bash
export HISIM_ENABLE=1
export HISIM_OUTPUT_MODE=BLOCKING
export HISIM_CONFIG_PATH="$(pwd)/hisim.json"

# Keep your existing DashLLM serving arguments and deployment configuration.
dashllm_cmd serving
```

Installation provides `hisim.pth`, which installs hooks at Python startup when
`HISIM_ENABLE=1`. Export the settings **before starting the service**, and make
sure workers and v6d children inherit them and can read the configuration path.
Changing the environment in an already-running interpreter does not rerun the
startup hook. Adding `src` to `PYTHONPATH` alone does not install the `.pth` file.

Set `HISIM_ENABLE=0` and restart the service to stop automatic hook installation.
Restore any simulation-specific engine settings when returning to real serving.

## Simulator configuration
`HISIM_CONFIG_PATH` points to a JSON file describing the simulated hardware
and predictor. It is separate from DashLLM's `DS_LLM_ENGINE_CONFIG`, which
configures the serving engine, including target parallelism.

A single-node configuration using `linear`:

```json
{
  "platform": {
    "accelerator": {
      "name": "l20b",
      "hbm_capacity_gb": 288
    }
  },
  "predictor": {
    "name": "linear",
    "base_ms": 2.0,
    "per_token_ms": 0.01
  }
}
```

These are example values, not calibrated performance estimates.

### Hardware and parallelism
- `platform.accelerator.name` selects a hardware preset; see
  [available presets](src/hisim/spec/accelerator/info.py). Custom devices require
  their capacity and bandwidth fields; see
  [configuration parsing](src/hisim/simulation/manager/config.py).
- `memory_*_bandwidth_gb` and `disk_*_bandwidth_gb` describe memory and disk
  bandwidth in GB/s; `num_device_per_node` describes the simulated node.
- Target parallelism comes from the engine arguments (`tensor_parallel_size`,
  `pipeline_parallel_size`, `data_parallel_size`, `enable_expert_parallel`).
  HiSim runs each DP rank with a single CPU executor, so TP/PP/EP do not
  launch corresponding workers.
- With `data_parallel_size > 1` (`BLOCKING` only), every step is synchronized
  across DP ranks like vLLM's model runner, and each rank predicts the same
  latency from the DP-wide batch. Each rank exports its stats to
  `$HISIM_OUTPUT_DIR/dp<rank>/`.
- `scheduler.physical_block_size` can override the modeled KV block size when
  matching a production cache layout. Leave it unset for framework-derived
  sizing, especially for hybrid attention/Mamba models.

### Latency predictor
`linear` is useful for integration checks and controlled load experiments:

```text
step_latency_ms = base_ms + per_token_ms × scheduled_tokens
```

Scheduled tokens are the sum of request `extend_length` values in the batch,
including prefill and decode tokens. Past KV length does not contribute. Both
coefficients default to zero and must be non-negative; empty batches have zero
latency. Set `per_token_ms=0` for constant latency per nonempty step. Linear
coefficients are explicit inputs, not inferred from the hardware configuration.

For calibrated prediction, replace the `predictor` section:
- `aiconfigurator`: requires compatible AIConfigurator profiling data and
  hardware/backend metadata. Options include `database_path`, `database_mode`
  (default `SILICON`), `prefill_scale_factor`, and `decode_scale_factor`. Set
  `scheduler.backend_name` and `backend_version` to the profiled backend.
- `ml`: requires `database_path` pointing to a compatible trained model bundle;
  `latency_scale` defaults to `1.0`. `database_path` may be an HTTP(S) URL; it is
  downloaded once to `/tmp/hisim/downloads` and reused until that copy is deleted,
  so publish updated bundles under a new URL. Bundles are loaded with `joblib`,
  which can execute code: only use URLs you trust.

With DP, `predictor.dp_batch_scope` selects the predictor input: `global`
(default) predicts the concatenated DP-wide batch, matching DP collector
training rows; `max_rank` predicts each rank's batch and keeps the slowest.
`predictor.dp_idle_step_ms` (default `0`) is the latency of steps where every
rank runs a dummy batch.

In `BLOCKING`, predictor CPU execution consumes the modeled time budget:
the remaining sleep is the modeled span minus prediction CPU time, clamped
at zero. DP coordination is outside this timing window and is not deducted.
This also applies to dummy ranks and async worker output. Recorded modeled
latencies stay unchanged; `prediction_cpu_latency` and `blocking_wait_latency`
expose the consumed CPU time and remaining wait. With `HISIM_RECORD_STATS=1`,
export statistics through the engine's profiling API before stopping workers.
`OFFLINE` virtual time uses the full modeled span.

Use `ml_dp` with a DP-aware trained bundle to retain per-rank batch composition:

```json
"predictor": {
  "name": "ml_dp",
  "database_path": "/path/to/dp_ml.joblib",
  "dp_batch_scope": "global",
  "sample_tokens_base_ms": 0.18
}
```

The `dp_v1` schema combines the global 18 features, seven load summaries, and
18 features per rank (61 features for DP2). Ranks are ordered by load so swapping
rank identities preserves the prediction. The bundle records `dp_size`,
`features`, `feature_schema`, and `label_kind`; incompatible bundles or rank
counts fail explicitly. `ml_dp` requires `dp_batch_scope=global`; it receives
the rank-separated batch directly, without concatenating it first.

The current label kind is `max_active_worker_execute_model_cuda_events_s`:
the maximum measured CUDA-event span among active workers. Idle-rank dummy
timing is not inferred from missing rows. Sampling is a separate calibrated
constant, and all-idle steps use `dp_idle_step_ms`. Calibrate for the target
hardware and TP/DP/EP configuration. Training data should cover small batches,
one active rank and imbalanced rank loads. Keep all TP copies of one step in
the same split, and reserve independent requests for end-to-end validation.
An artifact trained only on prefill does not validate decode-only prediction.

See [predictor selection and options](src/hisim/simulation/manager/config.py).
Restart serving processes after editing the JSON; configuration is cached in
each process.

### HTTP benchmarks with token traces

Use the serving image's vLLM benchmark through a streaming HTTP router. Install
the `benchmark` extra for `aiohttp`; keep the image's PAI-vLLM and DashServing.
Run the client and router with `HISIM_ENABLE=0` and `SIM_COLLECTOR_ENABLE=0`;
enable simulation only in the target serving processes.

```bash
python -m hisim.benchmark.router \
  --dashllm-address 127.0.0.1:8001 --port 18080

# In another terminal:
python -m hisim.benchmark.serve bench serve \
  --backend openai --model /path/to/local/model \
  --base-url http://127.0.0.1:18080 --endpoint /v1/completions \
  --dataset-name hisim --dataset-path /path/to/trace.jsonl \
  --num-prompts 2000 --timestamp-unit s --timestamp-scale 1 \
  --ignore-eos --save-result --save-detailed \
  --result-dir /tmp/bench-results --result-filename real.json \
  --trace-manifest /tmp/bench-results/real-trace.json
```

Each JSONL row contains `input_ids` (a nonempty integer array), `output_length`,
and `created_time` or `timestamp`. Optional `input_length` must match the array;
optional `request_id` is forwarded in the HTTP header. The loader selects the
first `num-prompts` nonempty rows, then sorts them by timestamp. It sends the
original IDs without decoding, tokenizing, or applying a chat template. Output
length is selected per request.

The adapter invokes vLLM's `bench serve` CLI and replaces only dataset loading
and arrival scheduling. HTTP requests, SSE parsing, TTFT and result statistics
remain native vLLM implementations. It requires a compatible vLLM exposing
`SampleRequest`, `get_samples`, and `get_request`; validate the wrapper when
upgrading the serving image.

Arrivals follow `(timestamp - first_timestamp) * timestamp_scale` using a
monotonic clock. `0.5` is twice as fast; `2` is twice as slow. Supported timestamp
units are `s`, `ms`, `us`, and `ns`; epoch seconds use `s`. No concurrency cap is
set by default. `--max-concurrency` is optional and can delay actual sends; do
not combine trace arrivals with `--request-rate` or ramp-up. The trace manifest
records the selected data hash, arrival span and sender lag so two runs can be
compared. Native readiness probes and optional warmups precede the trace clock
and can populate prefix caches; align cache state when comparing services.
Native result metadata identifies `arrival=trace` and includes the scale and
trace hash. The native `request_rate=inf` field is unused in this replay mode;
timestamps determine the actual arrival rate.

The router accepts `POST /v1/completions` with an integer-array `prompt`,
`stream=true`, `n=1` and a positive `max_tokens`. It uses DashLLM's async client
without a thread-pool concurrency limit. It emits a token-bearing SSE only
after upstream returns tokens, then a separate usage event and `[DONE]`.
`GET /health` reports active, completed, failed and peak-active requests;
`GET /v1/models` reports the configured model label. `--tokenizer` optionally
decodes response text; input token IDs are always passed unchanged. Server
capacity limits and queues still apply independently of the client settings.

### Remote PLE admission
Qwen4-style models served with `VLLM_PLE_OFFLOAD_BACKEND=remote_v6d` fetch PLE
rows for the whole prompt after external KV load and before the request is
schedulable. HiSim skips the PLE service and offload consumer and delays this
step by:

```text
admission_ms = base_ms + per_1k_tokens_ms × prompt_tokens / 1024
```

Prefix hits do not shorten it. Models without PLE layers (no `use_ple` /
`ple_layer_ids` in the text config) get no delay even when `remote_v6d` is set.
Defaults (`base_ms=5.56`, `per_1k_tokens_ms=0.383`)
were measured on L20B; override or disable them with an optional section:

```json
"ple": {"admission": {"enabled": true, "base_ms": 5.56, "per_1k_tokens_ms": 0.383}}
```

Per-step PLE compute is part of the forward and must come from the predictor,
e.g. an `ml` bundle trained on traces from the PLE-enabled deployment.

## DashLLM engine settings
Keep your usual model, transport, service-discovery, and v6d configuration.
Merge the following settings into your existing `DS_LLM_ENGINE_CONFIG` for a
simple simulation run; do not replace its connector settings:

```json
{
  "load_format": "dummy",
  "enforce_eager": true,
  "async_scheduling": false
}
```

Model configuration and tokenizer files must remain available. Inference hooks
skip real model execution, but framework model inspection and initialization
still run. Disable model preloading/staging through your serving image's
settings when it would otherwise copy real weights.

For cache tests, keep prefix caching and your v6d connector enabled. Input
length, context limits, and cache capacity must allow complete **physical** KV
blocks; short prompts may complete without storing anything in v6d. HiSim does
not provision Redis, trackers, sockets, or cache topology for your deployment.

On CPU-only hosts, ensure child workers have the intended device visibility
before bootstrap. Some DashServing launchers overwrite `CUDA_VISIBLE_DEVICES`;
HiSim's startup default does not override an existing value. Framework controls
remain distinct from simulator settings.

On ECI CPU nodes, HiSim uses the platform-injected `ALIYUN_ECI_ETH1_IP` when
v6d resolves its own hostname for tracker/P2P advertisements. Without this
variable (e.g. on Dash GPU nodes), resolution remains native. An invalid,
loopback, unspecified, multicast or link-local IPv4 raises a configuration error. This hook is scoped
to v6d peer modules: it does not edit `/etc/hosts`, change global DNS, probe
Redis, or rewrite KVCM subscriber reporter identities. No extra network
configuration is required in HiSim.

## Environment settings
Core simulator settings are defined in [src/hisim/env.py](src/hisim/env.py).

| Variable | Purpose | Default |
| --- | --- | --- |
| `HISIM_ENABLE` | Install hooks automatically at Python startup | Disabled |
| `HISIM_OUTPUT_MODE` | Use `BLOCKING` for live serving | `OFFLINE`; set explicitly |
| `HISIM_CONFIG_PATH` | Simulator JSON path, readable by every worker | Required unless `HISIM_CONFIG_JSON` is set |
| `HISIM_CONFIG_JSON` | Inline simulator JSON string; overrides `HISIM_CONFIG_PATH` | Unset |
| `HISIM_OUTPUT_DIR` | Directory for simulator outputs | `/tmp/hisim/output/`; the standalone worker wrapper uses `/tmp/hisim/<name>` |
| `HISIM_LOG_LEVEL` | Simulator logging level | `INFO` |
| `HISIM_RECORD_STATS` | Collect per-request/iteration simulator stats in `BLOCKING` mode | `0` |
| `HISIM_MAX_TOKENS` | Override request output length and ignore EOS for controlled tests | Unset: keep request limits |
| `HISIM_BW_PROFILE` | JSON calibration profile for v6d transfer-latency modeling | Unset: model disabled |
| `HISIM_DP_MAX_REQUESTS` | Per-rank request capacity of the DP synchronization payload; must match on every rank | Engine `max_num_seqs` |

Leave `HISIM_MAX_TOKENS` unset when testing the client's requested output lengths.
Leave `HISIM_DP_MAX_REQUESTS` unset unless DP ranks run with different
`max_num_seqs`; a step with more requests than the capacity fails.
Enable `HISIM_RECORD_STATS=1` only when internal statistics are needed; collection
adds overhead beyond the normal client-side measurements.

For cache-latency calibration, `HISIM_STORE_POLL_MS` and
`HISIM_STORE_RANK_SYNC_MS` override store-completion profile values.
`HISIM_SEG1_FLOOR_MS` and `HISIM_SEG1_PER_BLK_MS` set cross-node latency parameters;
overriding either replaces the profile pair, with the unspecified value set to
zero. See the [bandwidth model](src/hisim/simulation/hijack/v6d/bandwidth.py) for
the profile schema and applicability.

## End-to-end check
The [single-node smoke test](test/test_dashllm_single_node.py) uses the equivalent
DashLLM Python serving entrypoint, `BLOCKING` mode, and `linear` prediction:

```bash
python test/test_dashllm_single_node.py --model /path/to/local/model
```

Requires `redis-server`, `dashservingd`, and `v6d`. Random long-input requests
verify cache growth via v6d's `POST /objects/list` (metadata, not KV payload).
Services stop automatically; logs and snapshots remain in `tmp/dashllm-smoke-*`.
Latency estimates depend on calibration and include real serving overhead.
