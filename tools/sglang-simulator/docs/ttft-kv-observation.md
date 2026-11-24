# TTFT KV lifecycle observations — 2026-11-24

This is an observer-only change based on deployed `f7ec544`. It does not merge
fixcache behavior, replace stage models, alter native scheduling, change
DashScope services, or synchronize devices. The same helper is used by the
real `pp_v3` collector and CPU simulator. All times describe host observations.

## Locations

- `cache_boundaries.install_connector`: install per-instance query and connector
  scheduler observers after native construction.
- `kv_observation.install_query`: wrap the outer `_async_match` (fall back to
  `_async_match_impl` only when the outer method is absent) and
  `_commit_async_match`.
- `V6dObjectManager.async_lookup_results_many` / `async_fetch_selected_many`:
  context-scoped class helper wrappers split lookup from selected-object fetch.
  Calls without an active observed query pass through without events.
- `build_connector_meta`, `_submit_publication`, manager `seal`, and
  `update_connector_output`: observe native store IDs, publication RPCs and
  consumption of native worker job ACKs.
- `_free_blocks`: record actual scheduler block-release call, independently of
  logical request finish and connector delay-free decision.
- Native V6D/Raiden workers: after native construction/register, wrap existing
  handler `swap` and `get_finished` calls. Never poll handlers independently.
  Observe worker request terminals and native worker-meta export as well.
- The simulator composes observer construction into its existing worker hook;
  registration applies only the first matching class hook. The real collector
  registers a dedicated observer class hook for these workers.

## Fields and time boundaries

All new events carry `observation_contract=2`, `time_kind`, and
`device_completion_confirmed=false`; normal recorder envelope retains its
existing schema/process/clock/rank identity. No payload, KV values, block keys,
addresses, error strings or credentials are exported.

| Event | Correlation and boundary |
|---|---|
| `request.v6d_query_created/start/end/consume` | Existing event names preserved. Added `component_id`, `query_id`, native `generation`, hashed request ID, creation/start/end times and `query_linked`. `fetch_failed` and original hit fields remain. |
| `request.v6d_lookup_start/end` | `query_id` + `span_id`: naturally invoked group/batch lookup calls. |
| `request.v6d_fetch_start/end` | Same correlation for selected-object fetch. Its span includes native fetch/await overhead; not pure DMA. |
| `request.kv_store_declared` | Native scheduler `store_job_id` and request hash, when connector metadata is built. This is not worker submission or completion. |
| `request.kv_store_wait_declared` | Native request added to store-wait metadata. |
| `request.kv_worker_submit` | Native worker load/store/handoff call start/return, tokens when exposed. Not transfer completion. |
| `request.kv_job_submit` | Naturally called handler `swap`; local `native_job_id`, scheduler `store_job_id` if available, direction, request hash and call start/return. Submit return is not completion. |
| `request.kv_job_ready_observed` | Result already returned by native handler polling; preserves the result object and its iterator semantics. Completion may have happened earlier than this observation. |
| `request.kv_worker_meta_export` | Native worker-meta builder exports store results. No extra builder invocation. |
| `request.kv_worker_terminal` | Native worker get_finished returns request send/recv terminal. |
| `request.kv_job_ack_consumed` | Scheduler update sees worker job ACK. `ack_rank_count` is this incoming report's rank count, not cumulative or expected total. `job_retired` is only known when the native store-job map is available. |
| `request.kv_publication_submit/complete` | Publication submission and actual manager seal call/return; seal result is reported when boolean. Publication is distinct from data movement and final block release. |
| `request.kv_release_decision` | Native request_finished_all_groups return's `delay_free`, with native call span. |
| `request.kv_blocks_release` | Native scheduler _free_blocks call span; separate from request logical finish. |
| `request.*capabilities` | Installed methods/helper availability. Absence is unknown, never synthetic success. |
| `request.kv_observation_evicted` | Bounded 8192-entry correlation state evicted; completeness must be reported. |

Per-query timing decomposes as:

1. created -> start: waiting to begin the background coroutine;
2. start -> end: query execution, split by actual lookup/fetch helper spans;
3. end -> consume: completed background result awaiting scheduler consumption;
4. consume -> first selected: remaining readiness/scheduler wait, joined with
   existing scheduler membership events.

Do not sum overlapping helper spans. Native lookup can issue several queries.
Do not subtract monotonic timestamps across unrelated hosts. Cross-worker job
joins require request hash + logical store job ID + run/process/topology scope;
local handler job IDs alone are not global keys.

Transfer worker events follow the collector's existing representative PCP0
request-event policy; this change does not silently multiply all events by eight.
Scheduler aggregated ACKs provide the native job-consumption boundary. This is
not an all-rank device-completion trace.

## Safety and unknowns

Native arguments, return identity, exceptions and cancellation propagate.
On the request hot path observers create no task, wait, sleep, future.result, device event, synchronization,
extra lookup, poll or metadata-builder call. Normal writes use existing async
recorders. Correlation state is bounded; eviction is explicit. Worker recorders
reset after fork and close at process exit.

Generator completion results are deliberately not consumed to inspect them;
that path has no per-job poll event. Legacy workers without job metadata retain
request-level terminals/capability evidence. Bytes are not inferred from
unknown object layouts. Missing boundaries remain unknown.

No new platform API/gateway instrumentation is added. Grafana first_latency's
whole boundary is still not fully covered. Existing business->external->internal
alias events are exported, but cross-side pairing must be verified with real
requests, not presumed from matching sample counts.

## Export

`lifecycle_event_kinds.py` is the shared allowlist, including original private
request/identity/cache/query/phase events and all new job/publication events.
`export_lifecycle_window.py` retains the validated ASI whitelist-only reader,
source-path provenance, end sentinel, time/file bisection for truncated responses,
clock mapping, process sequence deduplication and rate limiting.

Changes from the historical exporter:

- Network/auth/command errors fail immediately instead of recursively bisecting.
- Missing inventory fails instead of waiting indefinitely.
- A receipt with an older event-selection digest cannot silently be reused.

Example on the authorized operator workstation, after requests have drained
and native recorder cuts/metadata are complete:

```bash
PP_RETRAIN_ROOT=/absolute/path/to/NEW_RUN \
PP_EXPORT_WORKSPACE=test \
PP_EXPORT_TRACE_BASE=/home/admin/NEW_COLLECTION/trace_v3 \
python tools/sglang-simulator/scripts/export_lifecycle_window.py \
  --case lifecycle-new --start START_EPOCH --end END_EPOCH
```

NEW_RUN must contain current `injection_complete.json` and
`injection/POD/metadata.json`, as produced by the existing inventory workflow.
Do not reuse old pods/metadata or cached receipts while source segments are
still being written. `full_selected_window` proves the persisted selection was
read, not that collection had no dropped events; also inspect recorder counters,
observer errors, capabilities and request inventory. CPU lifecycle JSONL uses
a different envelope and should be archived directly as before.

## Tests and rollout status

Tests cover async creation/execution/consumption separation, context isolation,
legacy/modern entrypoints, exact native argument/result/exception behavior,
registration idempotence, bounded state, generator non-consumption, native ACK
and release calls, real GNU grep export, truncation and network errors. Original
request/profile/grammar/AsyncLLM regression tests are retained.

This development change has not been deployed or accepted on real TPU hardware.
Before deployment, verify installed collector baseline and worker API, build a
fixed wheel, install in both actual runtime environments, restart only the
explicit test deployment through its established path, and run a few tagged
short/long/cache-hit requests. Accept only when new capabilities/events, request
alias joins, job chains, dropped/error counters and both exports are verified.
Keep code-only vs new-model accuracy experiments separate.
