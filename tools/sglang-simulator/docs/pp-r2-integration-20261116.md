# R2 source integration, 2026-11-16

This branch imports the runtime source shipped in the colleague's deployed R2 wheel:
`oss://kunlun-us-west/hisim/runtime/pp-lifecycle-r2-20261111/2f9f1f61d897/artifact.tar.gz`.
The upstream collection branch is still `aa73f1d7`; its two existing local commits
`11f3d50` and `94f3a0a` are retained. The artifact is the runtime source of record;
this is a source import, not a claim that a newer upstream Git commit exists.

The existing real collector, Prefill protocol boundaries, and native WORLD barrier
observations remain. Shared runtime modules come from the R2 wheel, including its
SGLang adapters needed by the original bootstrap; these are imported baseline code,
not new SGLang behavior or a change to the Prefill-only experiment scope.
The bootstrap selects collector or simulator using their existing environment flags.
Enabling both stops interpreter startup before either installs hooks. Vendored native
PP control files and their source manifest are included in built wheels.

The follow-up local commit retains the stage barrier timestamps and per-rank/padding
calibration already running in the personal test4 simulator. Predictors, runtime
configuration, local V6D isolation, and online deployments are not changed here.

Validation reuses the R2 CPU integration test and targeted request/native boundary
checks. CPU validation does not establish TPU timing accuracy or DRAM hit accuracy.


## Live source comparison

The colleague's revision 22 container and the personal simulator each contained
102 Python source files. Direct text comparison found only the personal changes
in `simulation/vllm/tpu_pp/stage_runtime.py`. Relative to the original R2 artifact,
both deployments also patch `simulation/vllm/engine_args.py` to preserve native
`create_engine_config` for DashLLM's PCP source inspection/backport, and patch
stage_runtime to preserve connector-state queue entries, seed the native stepper,
and apply the configured Prefill handoff delay. These existing live fixes are now
part of the source tree rather than installer-time text edits.

Combining the two distributions exposed a recorder-selection bug: directory
presence incorrectly selected the real collector even in simulator mode. Recorder
selection now follows SIM_COLLECTOR_ENABLE, preserving the simulator frontend ID
and first-output events when the collector package is also installed.

Do not point the old runtime bootstrap at the integrated wheel unchanged: the old
bootstrap applies the same source patches and validates their old preimages. A
future deployment should install this wheel directly and remove those duplicate
patch blocks. This change does not deploy or claim a new timing improvement.

Validation: 85 tests and 73 subtests passed using the existing vLLM 98dff2a source
and CPU container. The integration case completed 26 requests and checked prefix
reuse, abort, profiler drain/reset, request lifecycle export, and all four ranks'
barrier timestamps. Existing connector-state queue and timing environment values
were enabled (78.861 ms initial measured, 8.185 ms Prefill handoff).

## V6D query observation

The personal Aone branch adds per-request/generation coroutine creation, background start/end and scheduler result-consumption events. Creation is immediately before native runtime.submit, not a wire timestamp. The query execution span includes native cross-group lookup/fetch; it is not automatically KVCM service time. Native errors and cancellation propagate unchanged. Shared instance observers recurse through MultiConnector children on real and simulated schedulers. Validation: 18 focused tests passed, including delayed execution/consumption and failed observer cases. No latency model changes in this commit.

## Prefill input extent (2026-11-16)

Live r58 Raiden stage-3 producer admission removes the final prompt token before
prefix lookup (`on_new_request` -> `_maybe_truncate_prompt_for_handoff`). Decode
recomputes that token. The local CPU producer previously skipped admission and
therefore scheduled N tokens against the real Prefill N-1. The eight paired
requests at 16:12 all show this difference; member identity alone did not imply
matching chunk shapes.

Retain native producer admission after the existing remote-parameter rejection.
Copy the request's prompt list first because the in-process simulator can share
it with frontend accounting. This only matches Prefill work; it adds no Decode
or transfer simulation. Five focused checks cover native truncation/nontruncation,
a one-token prompt, repeated admission, unchanged frontend input, rejected remote
parameters, and the nonlocal path. This candidate is not part of the 18:50 baseline
(wheel 69dd301); deploy only after that comparison window finishes.
