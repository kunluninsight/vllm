# Finite Prefill pressure comparison

`tools/prefill_pressure.py` runs client concurrency 1, 16 and 64 for 600 seconds
per case, drains requests, then waits 300 seconds between cases. Full prompts
are unique; pairs share a controlled prefix. Case namespaces prevent cross-case
prefix reuse. Client concurrency is measured through SSE completion, so it is
not the engine's instantaneous concurrency.

The runner imports the already validated load generator via `--sender`. Pass
its `--target` and a one-line request-plan `--template` (containing `payload`),
a fresh `--output` directory and unique `--label`. Reuse the authorized
`DASHSCOPE_API_KEY` environment; never write it to experiment reports.

For the smoke gate, use `--concurrency 4 --duration 20 --max-requests 16 --gap 0`.
Check paired real/simulation request identities and real V6D query events
before the default 40-minute experiment. HTTP 400/401/403/404 stops further
admissions. Each case has a 20,000-request cap and a 120-second request timeout.
Save `manifest.json`, `requests.jsonl`, `summary.json` and root `cases.json`.

`tools/prefill_query_waits.py TRACE_ROOT RECEIPTS_CSV OUTPUT_DIR` consumes
`real/*.jsonl` and `sim/*.jsonl` plus CSV columns `sample` and `request_id`.
It joins business, external and engine identities, then splits admission to
first dispatch into query creation-to-start, execution, completion-to-scheduler
consumption, and time outside those observed intervals. Missing query hooks
are null, not zero. Query generation intervals are clipped to the first-dispatch
window and unioned to avoid double-counting. These are local host durations,
not device time or pure network latency. Incomplete queries remain explicit.

Validation on server 37 (2026-11-16): finite-runner fake-sender check completed
28 requests covering unique prompts, shared-prefix/case isolation, caps and
deadline/drain behavior. Synthetic query timeline yielded 10 ms start wait,
20 ms execution, 50 ms scheduler-consumption wait and 20 ms uncovered, preserving
zero hits. Live integration remains a separate gate; prepared scripts are not
experiment results.

### CPU transport admission follow-up (2026-11-16)

The 803804c live 16-request check still executed N tokens instead of the real producer's N-1: native on_new_request only truncates when Stage-3 transport is enabled. The local CPU connector now derives the truncation requirement from native Mamba/window cache specs and calls the native idempotent truncation helper after admission. It keeps frontend token accounting independent and does not enable a Decode peer or Stage-3 transfer. Seven regression cases cover CPU transport, existing native truncation, one-token prompts, full-attention prompts, remote-parameter rejection and non-local consumers. Live token-extent validation remains required.
