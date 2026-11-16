# Personal test4 P0 native-window candidate, 2026-11-17

Prefill only; KVCM disabled; local V6D unchanged. This candidate updates only the
P0 predictor artifact. Runtime code 3788f74, P1/P2/P3, host preparation, barrier,
and scheduler metadata settings remain fixed during its validation.

The full warm baseline shows a shape-dependent P0 prediction error: for batch
sizes 8 and 9, mean prediction minus native observation is +21.28 and +42.61 ms.
Low-batch predictions are already accurate. A 2.49 ms full-pipeline cycle error
accumulates into queue growth under sustained C64 traffic; final TTFT was never
used as a training target.

The candidate keeps the original 300 GBR trees and adds 60 depth-2 trees with
learning_rate=.05, subsample=.8, min_samples_leaf=12. Training labels are 1026 P0
native spans from phase92-c4 and phase92-c64short. Labels run from forward entry
to local drain observation; they are not isolated device timings. Cold and long
observations are retained. Existing PCP-padding offsets remain unchanged.

Independent full warm holdout, 1705 batches: MAE 6.650 -> 2.584 ms, signed bias
+5.683 -> +1.037 ms, P95 absolute error 32.681 -> 7.598 ms. Independent low-pressure
meta-b176-c4, 24 batches: MAE .694 -> 3.963 ms and bias +.089 -> +3.951 ms. This low
pressure regression is explicit and requires fresh end-to-end C1 validation.
One batch-size-11 warm observation has candidate error -46.11 ms; it remains in
the metrics. Do not claim acceptance from component validation alone.

Evidence on server 37: qwen38-tpu-pp/work/pressure-1116/
- fit_p0_native.py (exact fixed fit and serialized-runtime check)
- candidate-p0-native-1117/validation.json and predictions.csv
- phase92-c64short/calibration/p0-full.jsonl (1003 batches)
- phase92-c64warm/calibration/p0-full.jsonl (1705 held-out batches)

The model pickle contains native_recalibration provenance and all split metrics.
Keep the original P0 artifact for rollback. Future acceptance needs fresh mirrored
requests, stable queue behavior, and aligned throughput/cache counters.
