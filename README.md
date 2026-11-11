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
`HISIM_CONFIG_PATH` points to a JSON file describing the simulated hardware,
predictor, and target parallelism. It is separate from DashLLM's
`DS_LLM_ENGINE_CONFIG`, which configures the serving engine.

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
  },
  "scheduler": {
    "tp_size": 1,
    "ep_size": 1
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
- `scheduler.tp_size`, `ep_size`, and `dp_size` describe target parallelism for
  simulation. HiSim forces vLLM engine parallelism to one and uses a single
  executor; target parallelism does not launch corresponding GPU workers.
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

See [predictor selection and options](src/hisim/simulation/manager/config.py).
Restart serving processes after editing the JSON; configuration is cached in
each process.

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

## Environment settings
Simulator settings are defined in [src/hisim/env.py](src/hisim/env.py).

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

Leave `HISIM_MAX_TOKENS` unset when testing the client's requested output lengths.
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
