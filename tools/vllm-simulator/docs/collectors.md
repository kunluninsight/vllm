# Collect GPU and frontend timings

Enable `simulator_collector` on a native server with the target model and deployment flags.
The worker supports TP with PP=DP=PCP=DCP=1 and speculative decoding disabled.

```bash
VLLM_PLUGINS=simulator_collector \
  .venv/bin/python -m vllm.entrypoints.cli.main serve "$SIM_MODEL" \
  --worker-cls vllm_simulator.collector.worker.CollectorWorker \
  --additional-config '{"simulator":{"collector":{"output_dir":"/path/to/exports","gzip":true}}}'
```

Use `vllm bench serve --profile`, or POST `/start_profile` and `/stop_profile` on
an idle server around each window. Stop synchronizes outstanding CUDA events
and replaces the previous files. Archive them between runs. Rows are buffered
in memory until stop; shutdown discards an unfinished window.

Each worker writes `rank{rank}.schedule_batch.jsonl.gz` under `output_dir`
(default `.`). Set `gzip: false` for plain JSONL. Rows contain a window-local
step ID, wall-clock start time, forward mode, scheduled token count, forward
and sampling durations in seconds, and `[request_id, extend_length, past_kv_length,
output_length]` per request. Event recording adds overhead and measures GPU
stream time around worker RPCs. For TP, align steps across ranks and take both
durations from the rank with the largest forward-plus-sampling time.

```python
from vllm_simulator.time_predictor.ml import MLTimePredictor
from vllm_simulator.time_predictor.replay import ReplayTimePredictor

trace = "rank0.schedule_batch.jsonl.gz"  # Single-GPU calibration
ReplayTimePredictor.build_table_from_schedule_batch(trace, "replay.json")
MLTimePredictor.build_model_from_schedule_batch(trace, "ml.joblib")
```

Set `additional_config.simulator.predictor` to `{"name":"ml","database_path":"ml.joblib"}`
or `{"name":"replay","database_path":"replay.json"}`. ML requires the plugin's `ml`
extra. Replay averages exact shape matches; unseen shapes take zero time by
default, or use `miss_strategy: knn` for interpolation. Validate on separate traffic.

For frontend calibration use a single API process, `request_timeline: true` in
the collector config, `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`, unique request
IDs and `--save-detailed`. On one host, match the client's monotonic `start_times`,
`ttfts` and `latencies` to `queued_at`, `first_token_at` and `last_token_at` in
`requests.jsonl.gz`. Fit ingress and first-chunk delivery against prompt length;
use `queued_at - start_time` and `start_time + ttft - first_token_at`, respectively.
For later-chunk delivery use `start_time + latency - last_token_at` from multi-token
requests. Convert seconds to the units in the simulator client profile.
