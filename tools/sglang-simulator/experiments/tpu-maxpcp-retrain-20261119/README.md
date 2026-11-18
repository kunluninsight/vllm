# Max-PCP direct stage models — 2026-11-19

This experiment replaces the old stage-specific padding offsets with direct
models trained on complete eight-PCP observations. It does not alter deployment
or runtime configuration. The parent rollout separately removes runtime offsets
and sets `barrier_overhead_s=0.00058`.

## Label and feature contract

For one exact `[collection, scheduler process, schedule id, side]` context and
chunk, `target_ms = max_i((observed_mono_ns_i-start_mono_ns_i)/1e6)` over PCP 0–7.
P0–P2 use `native.local_drain.ready`; P3 uses `native.output_ready_observed`.
These remain effective host-observed windows, not isolated device service time.
No hand-written offset is added to or subtracted from the target.

The loader rejects missing ranks/steps, duplicate step contexts, different rank
identities, different contexts/chunks across PCP or PP stages, collection mismatch,
failed native calls, invalid intervals and non-finite features. Same numeric step
IDs from different sessions must be in different manifest datasets.

Each rank's interval uses that process's own monotonic timestamps. Start/end
skew and the earliest-start/latest-end envelope are diagnostic columns only,
computed only when all eight metadata files agree on clock domain, clock source,
boot ID and time namespace. The envelope never becomes the target.

Feature order comes directly from `stage_model.py`: P0–P2 have 22 features,
P3 has 26. Concurrency, TTFT, dataset identity and timestamps are excluded from
the feature vector. No old model supplies labels.

## Input and commands

Copy `manifest.example.json` outside the checkout and add the new low-load
dataset entry. Every entry contains `name`, `features_csv` and four `stage_dirs`.
Each stage directory contains `pcp{0..7}-events.jsonl` and
`pcp{0..7}-metadata.json`. The feature CSV uses the existing ramp feature schema
(`step`, `concurrency`, 26 feature columns), with an optional `case` column.
An optional `contexts_json` uses the existing `cases.json` structure and validates
that each feature row refers to the intended full context.

Run in the established Python 3.12 container with:

```sh
export PYTHONPATH=/host/qwen38-tpu-pp/work/cadence-replay-1113/stage-deps:/host/qwen38-tpu-pp/work/pp-lifecycle-20261112/real-barrier-1113/tools/sglang-simulator/src
python3 retrain_maxpcp.py --manifest /path/to/manifest.json --output /path/to/aggregation
python3 -m unittest -v test_retrain_maxpcp.py
# Only after adding complete low-load real observations:
python3 retrain_maxpcp.py --manifest /path/to/manifest.json --output /path/to/candidate --train
```

Default execution only aggregates. Training additionally requires at least 20
actual training observations per stage per global padding (2048/4096/8192).
`--min-train-per-padding` makes that guard explicit and reproducible. The old
four pressure cases alone have only 6/8 small-padding steps and are insufficient.

## Training and validation

Within each dataset/case/stage, order by PCP0 forward-start timestamp, reserve
the final 20% as contiguous holdout, and exclude the four immediately preceding
steps from fitting. Gap rows remain in the exports and are reported separately.
This is an observed-session temporal holdout; it does not claim unseen-pod or
unseen-session generalization. A fresh online four-case run is the final external
validation and must not be used to tune model parameters.

Each stage has one `GradientBoostingRegressor` predicting seconds directly:
300 trees, learning rate 0.05, depth 6, minimum leaf 12, subsample 0.8,
`loss='huber'`, `alpha=0.9`, seed 42. The tree count/depth follow the existing
baseline scale; leaf regularization and Huber loss limit isolated unexplained
tails' influence. All valid latency outliers remain in the data; none are capped,
removed or replaced. This is a robust conditional central predictor, not a tail
distribution sampler.

`labels.csv` retains all local durations and timestamps, maximum PCP, PCP0 gap,
clock qualification, envelope and split. `outliers.csv` is a diagnostic subset:
within dataset/case/stage/padding, absolute target deviation above
`max(20 ms, 6×1.4826×MAD)`; rank-gap or start-skew over 20 ms are also flagged.
The flags never filter training or evaluation. `coverage.csv` exposes sparse bins.

Training writes four compatible bundles plus `predictions.csv`, `metrics.csv`,
and `training.json`. Metrics include mean bias, MAE, P95 absolute error, MAPE and
P95 APE by stage/split and case, padding, batch size, and outlier flag. Exported
model reload is checked against training-library predictions. The existing
loader clips each model output to nonnegative; there is a single total-duration
model, so no negative residual is silently lost.

No commit, push, model deployment or final fit is part of script preparation.
