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
  .venv/bin/python -m vllm.entrypoints.cli.main serve "$SIM_MODEL" \
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

## Fit a predictor

Use a calibration trace from the target model, hardware, parallelism and
scheduler configuration. For a single GPU, build either a lookup table or
a regression model from its export. Install the plugin's `ml` extra for the
regression model.

```python
from vllm_simulator.time_predictor.ml import MLTimePredictor
from vllm_simulator.time_predictor.replay import ReplayTimePredictor

trace = "rank0.schedule_batch.jsonl.gz"
ReplayTimePredictor.build_table_from_schedule_batch(trace, "replay.json")
MLTimePredictor.build_model_from_schedule_batch(trace, "ml.joblib")
```

Set `predictor` to `{"name": "replay", "database_path": "replay.json"}` or
`{"name": "ml", "database_path": "ml.joblib"}` under
`additional_config.simulator`. Replay averages matching step shapes. For
unseen shapes it defaults to zero seconds. Set `miss_strategy` to `knn` to
interpolate from neighboring shapes. Validate request-level TTFT and TPOT on
a separate trace with the predictor held fixed.
