# ML and Replay GPU collector

`simulator_collector` is an opt-in vLLM endpoint plugin paired with
`vllm_simulator.collector.worker.CollectorWorker`, a native GPU Worker subclass.
It exports measured execution for ML and Replay predictors without enabling the
simulator platform or modifying vLLM classes. Tensor parallelism is supported;
PP, DP, context parallelism and speculative decoding are unsupported here.

## Enable collection

Install the simulator package into the vLLM environment. Set `SIM_MODEL` to the
requested model ID or local directory, and add the native serving options for
that model and topology:

```bash
VLLM_PLUGINS=simulator_collector \
  python -m vllm.entrypoints.cli.main serve "$SIM_MODEL" \
  --worker-cls vllm_simulator.collector.worker.CollectorWorker \
  --additional-config '{"simulator":{"collector":{"output_dir":"/path/to/exports","gzip":true}}}'
```

POST `/start_profile` and `/stop_profile` while the engine is idle to delimit
collection windows, or use the native `vllm bench serve --profile` flag.
No `--profiler-config` is required; these routes control collection. Only steps
between start and stop are recorded. Start resets the in-memory buffer without
creating or changing export files. Stop waits for recorded CUDA events and
exports the buffered rows, replacing the previous export. Archive completed
exports between runs. Shutdown without stop discards the unfinished window.
Enable `simulator` separately when replaying on CPU.

## Export contract

Collection parameters live in `additional_config.simulator.collector` and are
passed to every worker through vLLM's engine configuration:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `output_dir` | `"."` | Export directory, relative to each worker's current directory. |
| `gzip` | `true` | Write gzip-compressed JSONL; `false` writes plain JSONL. |

Use an export directory accessible to all workers. Each worker owns its
`rank{rank}.schedule_batch.jsonl.gz` (or `.jsonl`) file and resets `step_id` to
zero at the start of a window.

Each row contains a step ID, wall-clock start timestamp, scheduled token total,
forward mode, `iter_latency` (forward seconds), `sample_tokens_latency`
(sampling seconds), and `requests` tuples:

```text
[full_request_id, extend_length, past_kv_length, output_length]
```

The scheduled lengths are captured before model execution. Request token IDs
are not exported. CUDA events surround the native Worker RPCs. Subsequent
steps drain completed events into memory without synchronization; stopping
synchronizes outstanding end events and writes the file. Memory usage grows
with the number of recorded steps until stop. Native return values and async
output objects pass through unchanged. GPU stream durations exclude some
CPU/HTTP overhead; event recording and buffering also have collection cost.

## Calibrate and validate on separate runs

Keep the native model, device, parallelism, precision, cache geometry and
scheduler settings fixed. Choose training traffic that covers the decode batch
sizes and prefill lengths expected at deployment. A high request rate can
produce few small batches even when it generates many tokens. Compare step
counts by phase and batch size across training and evaluation.

Collect additional calibration traffic with its own seed. In the
`simulator-accuracy` workflow, pass its case directory to
`collector_accuracy.py evaluate --calibration-case DIR --evaluation-dir NEW_DIR`.
The target rate must be absent from the calibration case. The evaluator checks
native configurations and request IDs, then records the training/target file
hashes alongside the predictor. Compare baseline and calibrated predictions on
the same client arrivals and token lengths; cross-check other load levels and a
fresh request trace with the fitted predictor held fixed.

Archive observations (raw TP traces, native snapshots, GPU clock samples,
bundles, per-request timings and reports) in a new directory below
`var/plugin_accuracy/` for each experiment.

ML training accepts `loss="absolute_error"` in
`MLTimePredictor.build_model_from_schedule_batch`, or `--ml-loss absolute_error`
in the accuracy workflow. It estimates the conditional median forward time for
long-tailed measurements. The default squared-error fit estimates the conditional
mean. Compare both at the request level: a better typical step need not predict
end-to-end latency better under queueing. Saved bundles use the same runtime
predictor and feature schema.

Pool batch samples from multiple calibration runs in one training call:

```python
bundle = MLTimePredictor.build_model_from_schedule_batch(
    ["run1/schedule_batch.jsonl.gz", "run2/schedule_batch.jsonl.gz"],
    "ml.joblib",
)
print(bundle["training_summary"])
```

The default fits each phase once on the pooled samples and saves one bundle.
It does not train a predictor per input file or change weights by source.
Concatenating the files first gives the same default fit; passing separate
paths preserves per-run sample counts and optional validation groups.

Training logs and saves `training_summary`: scheduled and skipped empty steps
per source, prefill/decode counts and fractions, and min/p50/p90/max for batch
size, scheduled tokens, maximum past KV length and forward latency. Joint
coverage groups batch size, scheduled tokens and maximum past KV length into
power-of-two ranges (zero has its own range). It reports observed regions with
fewer than five samples and up to five examples. These are descriptive counts,
not an automatic uniformity or accuracy test: absent combinations may matter
for the intended deployment even when each individual dimension looks covered.

Merge only compatible model, device, precision and parallel configurations,
using healthy GPUs. TP peer exports describe the same step; align them and
select the critical rank before pooling runs, rather than treating every rank
as an independent batch. The experimental DP joint predictor additionally
requires aligned batches from both DP groups; concatenating DP rank files does
not supply that joint feature schema.

For optional per-phase model selection, pass `cross_validate=True` and inspect
`bundle["selection"]`:

Supply one merged critical-rank export per calibration window. Selection holds
out entire files (up to five folds), or five contiguous folds for one file. It
compares the requested tree loss, absolute-error trees, relative-error-weighted
trees with smaller leaves, and a scaled Huber regression, scoring mean absolute
percentage error separately for prefill and decode. Phases with fewer than five steps keep the
requested tree model. Selected models are refitted on all calibration steps;
the saved bundle records their validation scores. Sampling remains separately
configured by default. The default training behavior is unchanged.

Set `balance_windows=True` with cross-validation to give each calibration file
an equal share of the validation score. This prevents a long decode window from
outweighing short prefill or high-concurrency windows. It also compares extra
trees and a Huber fit using batch size, scheduled tokens, past tokens and
attention-work terms. Keep the target workload outside all calibration files.

Set `fit_sampling=True` to learn RPC-2 latency from each row's
`sample_tokens_latency` using the same batch features. This distinguishes batches
with equal scheduled tokens but different request counts or past KV lengths.
The saved sampling model replaces configured sampling compensation when
predicting a step. Its cross-validation score normalizes sampling error by the
full measured step duration. Existing bundles retain their configured sampling
compensation.

Model selection cannot replace missing calibration coverage. Include short
prefills and small decode batches, then freeze the bundle and compare request
TTFT and TPOT on independent traffic. Trees predict constant values outside
their training range; linear extrapolation can also fail across kernel changes.
Use these validation results to choose a deployment bundle rather than fitting
an offset to target latencies.

## Align requests and steps

This is an opt-in debug facility. Start a single API frontend with
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`, set
`additional_config.simulator.collector.request_timeline` to `true`, and keep
request stats enabled. Use `n=1`, unique request IDs throughout each collection
window, and idle engines at start/stop. The current vLLM environment switch is
marked for future removal; the collector rejects timeline startup without it.
Normal serving and simulation do not require the switch.

The collector observes existing EngineCore output events on this frontend's
output processor instance and joins the unchanged finished-request stats by ID.
It does not add fields to vLLM's core stats structures.
The same idle start/stop window writes one `requests.jsonl.gz` (or plain JSONL)
beside the rank exports. Each completed generation has:

- `request_id`: the exact internal ID in every rank's step tuples.
  `external_request_id` is the frontend's ID for that generation.
- `queued_at`, `first_scheduled_at`, `first_token_at`, `last_token_at`: seconds
  on the EngineCore monotonic clock; unavailable events are null. Token times
  denote EngineCore outputs containing tokens. A single token has equal first
  and last times. These are distinct from client receipt and CUDA completion.
- Engine index, prompt and generated token counts, and finish reason.

Use stable client IDs and verify a one-to-one join to the benchmark, including
input/output lengths. Finish benchmark traffic before stopping. The worker's
`worker_start_monotonic_s` marks entry to a scheduled worker call; CUDA event
durations are separate. Compare absolute times only on verified common clocks
(e.g., Linux processes sharing boot ID, time namespace and CLOCK_MONOTONIC).

For CPU replay, set `additional_config.simulator.step_trace_dir` to a fresh
directory to record each nonempty worker batch in `dp{rank}.steps.jsonl`.
It contains internal IDs, extend/past lengths and start/end on the simulator
clock (virtual seconds in NON_BLOCKING mode). Join to per-request simulation
results using the unique IDs supplied by the client. For this debug workflow,
set `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` on the simulator too and use
`n=1` with one API frontend. No internal ID field is added to simulation stats.
Compare sorted (request identity, extend, past)
tuples: after the first mismatch, equal step indices may describe different
work. Price the native shapes separately from free-running replay.

Replay tables can use `miss_strategy: "knn_work"` for workloads with variable
prompt lengths or chunked prefill. It indexes one-token decode batches separately
from batches with multi-token extensions and uses the number of causal
query–key pairs to account for both newly scheduled tokens and cached context.
On a decode-only miss, forward uses the matched batch shapes and sampling uses
the median for the batch size, with linear interpolation between observed sizes.
This avoids spreading transfer-related sampling timing spikes to batches with
similar context lengths.
Exact hits still use the measured table entry.

For formal prefill validation on GLM TP4, default to 150 requests per
measurement window in the accuracy workflow, with independent seeds
for calibration and validation. Target 8192–16384 computed prompt tokens per
request and reserve context space for the one output token. Launch native
serving with deployment defaults; specify only the model, required topology,
endpoint and instrumentation. If the model's default context cannot fit,
record the startup error and use the documented deployment context limit
without changing the step budget. Snapshot the healthy server's resolved
`/server_info?config_format=json` for replay. Keep prefix caching as deployed:
use independent prompt IDs without shared first blocks and replay those exact
token sequences. Check the worker traces: sum the scheduled extend lengths
over all chunks of each request on the merged TP trace, and compare the result with
the benchmark prompt length. Let the native scheduler select its step budget.

## Frontend timing

For frontend calibration use a single API process, `request_timeline: true` in
the collector config, `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`, unique request
IDs and `--save-detailed`. On one host, match the client's monotonic `start_times`,
`ttfts` and `latencies` to `queued_at`, `first_token_at` and `last_token_at` in
`requests.jsonl.gz`. Fit ingress and first-chunk delivery against prompt length;
use `queued_at - start_time` and `start_time + ttft - first_token_at`, respectively.
For later-chunk delivery use `start_time + latency - last_token_at` from multi-token
requests. Convert seconds to the units in the simulator client profile.
