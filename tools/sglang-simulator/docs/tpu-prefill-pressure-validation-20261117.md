# test4 Prefill pressure validation, 2026-11-17

The fixed candidate completed C64, C1, and C16 with 600 seconds of unique full
prompts per case and 300 seconds of idle time after each case drained. Shared
prefixes are intentional. The 17,054 formal requests all succeeded and all
paired by request ID with the mirrored simulator. A separate 16-request smoke
batch passed first. This is a warm-runtime Prefill experiment, not cold-start
or Decode validation.

Real: collector 92a8db0, revision 10. Simulator: wheel 3788f74, P0 model
41a7705, P3 residual model recorded by 7ce9ff0, revision 19. KVCM is disabled;
local V6D is configured. All cases use the same metadata and stage parameters.
No model was changed during the formal run.

| Client concurrency | Requests | Real/sim engine mean ms | Mean error | P95 error | Monitor Input TPS error | Monitor prefix relative error |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 452 | 434.178 / 430.381 | -0.874% | -1.463% | +0.176% | +0.503% |
| 16 | 5201 | 851.202 / 877.468 | +3.086% | -0.845% | +0.065% | +0.232% |
| 64 | 11401 | 2242.493 / 2089.111 | -6.840% | -0.866% | -0.033% | +0.512% |

Engine TTFT spans scheduler admission through consumption of the first Prefill
result, including queueing. Its whole-case mean and P95 are within 10% in all
three cases. Monitor TPS/prefix use the existing rule requiring the entire
rate(...[1m]) lookback to lie within admission: eight minutes and 32 native
points per side and metric. Raw nine-minute summaries are retained separately.
Whole-case input and engine prefix totals match exactly on both sides.
The mirrored Input TPS comparison is not an independent maximum-capacity test.

## Component evidence

The metadata budget executes at the native scheduler metadata boundary and
subtracts native CPU time already spent. The former 8.185 ms result-eligibility
wait is disabled. P0 and P3 calibration uses native component labels rather
than TTFT or concurrency as training inputs. See the preceding calibration
notes for training data and independent holdouts.

The prior P3-r1 C64 run had mean error -27.822%; the new run is -6.840%.
Full-pipeline dispatch intervals are 177.255 ms real and 177.679 ms sim.
First-dispatch waiting is 1468.3/1305.1 ms; last-dispatch to ready is
706.0/715.3 ms. A 96-batch, full-window sample gives prediction-minus-observation
biases of +1.503, -1.071, +0.339, +0.520 ms for P0 through P3.

For 5,287 requests with exactly one query on both sides, admission to query
creation is 1175.889/1018.691 ms, query execution is 15.698/3.678 ms, and
query-ready to consumption is 266.893/276.877 ms. Most residual first-dispatch
waiting precedes creation of that request's query. It is not a KVCM-query
interval. All 10,612 real and 10,547 simulated C64 V6D queries missed.
V6D storage planning/metadata still executes on misses.

## Limits retained in the result

- C64 minute errors range from -21.274% to +8.626%; three of ten minutes are
  outside +/-10%. Paired-request MAPE is 1.720%, 16.977%, 13.724% at C1/C16/C64.
  Passing means and P95 do not establish a 10% per-request bound.
- C64 local interarrival gaps below 2 ms occur in 40.781% of real arrivals and
  11.421% of sim arrivals. 23.079% of adjacent real arrivals reverse order in
  sim. This describes changed arrival shape, not a measured causal allocation
  of the remaining TTFT error. Twelve first-wave requests form six shared-prefix
  pairs that exchange the cache-producing request; pair totals are unchanged.
- External V6D query hits were not exercised. These data establish mainly HBM
  prefix behavior, not DRAM-hit loading accuracy.
- Raw deployment-monitor first-latency errors are -18.499%, -9.144%, -10.983%
  at C1/C16/C64. The 20-request stratified samples per case put real
  deployment-minus-backend duration at 134.935/141.099/149.358 ms versus
  28.771/30.296/47.788 ms in sim. These nested durations include platform work;
  they are not one-way inter-region network measurements.
- Outer-duration versus prompt-length descriptive R-squared values are
  .03695/.00157/.00272 in those samples. Three pressure levels do not establish
  a linear network-delay model. No outer delay was added to engine TTFT.

## Reproduction and evidence

Server 37: qwen38-tpu-pp/work/pressure-1116/, cases p3r2-main-c{1,16,64},
full mry-p3r2-main-c*.tar.gz archives, p3r2-summary/ outputs, and
paired-request-evidence.zip. Source real exports remain alongside their
exact-line merged views. The desktop diagnostic directory is
bailian/diagnostics/qwen38-tpu-pp-pressure-20261116/, including the three raw
monitor exports, cases-p3r2-main.json, monitor_p3r2_summary.py, and p3r2-main/.

Runtime parameter values and deployed artifacts remain unchanged by this
validation documentation commit. The prior revision-18 simulator spec is
retained privately as the rollback input.

The same 96 selected launches also retain PCP0 handoff boundaries for all four
PP ranks. Real barrier-call means are P0 6.386, P1 31.209, P2 31.961, and
P3 19.283 ms. Barrier return to payload submission means are .759/.640/.636/.324
ms. These are overlapping local waits and host submission spans, not additive
cross-rank costs or isolated network/device times. Tick-mode barrier retention
covers PCP0; it does not establish all eight PCPs' entry/exit alignment.
