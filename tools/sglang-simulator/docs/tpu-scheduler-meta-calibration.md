# TPU Prefill scheduler metadata budget

The optional `predictor.scheduler_meta` accounts for host time at the native
`Scheduler._build_kv_connector_meta` call. Its fields are `base_ms`,
`cached_request_ms`, and `query_token_ms`. The target is their linear sum using
the scheduled query tokens and scheduled cached-request count. Native metadata
creation executes first; only the positive remainder is waited on. Empty steps
and native failures retain their original behavior. No cache hits, allocation
results, PP barrier dependencies, or request state are fabricated.

Set `HISIM_TPU_PP_PREFILL_HANDOFF_MS=0` when enabling this model. The previous
post-result eligibility delay covered a different endpoint and cannot be kept
as an additional metadata budget. `scheduler.meta_budget` records native return,
modeled target, and final return separately; nested native phase events still
measure the unpadded native calls.

## Test4 candidate, 2026-11-17

This is a diagnostic candidate, not an accepted accuracy result. KVCM is disabled,
local V6D is enabled, and only Prefill is modeled. Calibration uses the first
668 of 1003 real nonempty batches from `phase92-c64short`; the final 335 batches
are held out chronologically. The measured label is connector metadata wall
time, never end-to-end TTFT. The three-term nonnegative fit is:

- base_ms: 2.0679355861920063
- cached_request_ms: 2.220580826269437
- query_token_ms: 0.0012872479505260146

Held-out mean label 14.301 ms; MAE 4.504 ms; mean prediction bias -0.467 ms.
The real four-node V6D metadata path and simulated local control plane differ;
this budget approximates their aggregate host cost, not isolated RPC latency.
It does not explain cold P3 output outliers. Validate on fresh mirrored traffic
at multiple pressures before treating it as accepted. Keep already measured
stage/barrier and network intervals separate.

Evidence: server 37 `qwen38-tpu-pp/work/pressure-1116/phase92-c64short/analysis/`:
`scheduler-meta-fit-full.csv`, `scheduler-native-phases.csv`,
`scheduler-native-methods.csv`, and `stage-calibration-by-time.csv`.
