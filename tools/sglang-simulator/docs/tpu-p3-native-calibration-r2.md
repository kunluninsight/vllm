# Personal test4 P3 native-window residual, 2026-11-17

This candidate changes only the P3 artifact on top of the existing P3 residual.
The active P0 candidate and metadata accounting remain fixed. Prefill only,
KVCM disabled and local V6D unchanged. This is component calibration, not
end-to-end acceptance; the current 40-minute run is kept on its original model.

Training uses 96 phase92-c64short batches plus 24 meta-b176-c4 batches. The label
is native forward-dispatch entry to output-ready observation, including the
native output path, not isolated device execution. No request TTFT or concurrency
value is a training feature. All observations, including cold >1s tails, remain
in the data and reported error. Huber loss downweights them during fitting.

The additional positive residual uses global padded tokens, actual batch size,
and prompt-complete request count. A sklearn ColumnTransformer selects these
three existing features, StandardScaler normalizes them, and HuberRegressor
(epsilon=1.35, alpha=.01, max_iter=1000) predicts seconds. The existing StageModel
clips each submodel below zero. Existing padding offsets are unchanged.

Independent holdouts (prediction minus observed, ms):

| Window | Batches | Old/new MAE | Old/new bias | Old/new P95 absolute |
| --- | ---: | ---: | ---: | ---: |
| phase92-c64warm | 96 | 6.396 / 2.560 | -5.836 / -0.595 | 12.362 / 8.049 |
| meta378-c64short | 96 | 5.639 / 1.953 | -5.057 / +0.282 | 11.708 / 5.378 |
| p0r1-c4 | 23 | 1.484 / 0.923 | -1.396 / -0.529 | 4.175 / 2.710 |

The source short-run outliers remain: training MAE is still 20.429 ms, bias
-18.338 ms, and P95 absolute 91.481 ms. This candidate does not explain those
waits or turn them into a regular device cost. A reloaded artifact reproduces
the candidate predictions within 1e-8 ms for every recorded row.

Exact reproducible fit and evidence on server 37 under
qwen38-tpu-pp/work/pressure-1116/: probe_p3_native_r2.py;
candidate-p3-native-r2-1117/{validation.json,predictions.csv}; and each source
case's analysis/p3-calibration-features.csv and calibration/node3.jsonl.
The pickle embeds native_recalibration_r2 provenance and metrics. Deployment
requires a fresh small mirrored batch followed by sustained pressure; retain
P3 r1 as the rollback artifact. Never claim the final accuracy target from the
component holdouts alone.
