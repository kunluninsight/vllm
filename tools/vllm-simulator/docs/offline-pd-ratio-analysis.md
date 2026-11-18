# Offline P:D Ratio Analysis

This example estimates P:D allocations for a fixed tokenized workload using
round-robin routing and independent `NON_BLOCKING` replays. The executable is
[`offline_pd_ratio_sweep.py`](../examples/offline_pd_ratio_sweep.py).

## Why the stages can be replayed separately

Under this model, each P instance's completion times depend only on its input
trace, configuration and local scheduling/KV state. D load and completion do
not change P admission or execution. P completion events can therefore be
computed first and used as a fixed input trace for D. There is no feedback
from D that would require rerunning or synchronizing the P engines.

The round-robin assignments also need no live load queries. P routing is
computed from original arrival order, and D routing is computed from P
completion order. Once these assignments are fixed, each instance's queue and
KV pool evolve independently. Supply the complete partition up front so its
native scheduler can replay arrivals, batching and KV pressure in virtual time.

For each request, the dependency between phases is:

```text
D_created_time = P_finished_time + transfer_delay
D_first_scheduled_time >= D_created_time
```

Both phases use the same time origin. Running all P replays before any D
replay in wall time does not make D wait until every P has finished in virtual
time. For example, with zero transfer delay:

| Request | Original arrival (s) | P finished (s) | D arrival (s) | D finished (s) |
| --- | --- | --- | --- | --- |
| r0 | 0.000 | 0.010 | 0.010 | 0.014 |
| r1 | 0.001 | 0.030 | 0.030 | 0.034 |

Here D processes r0 while P is still processing r1, even though the script
calculates the entire P phase first. Do not rebase each partition to its own
first arrival or add the wall-clock durations of the two replay phases.

This decomposition preserves the modeled causal dependencies. Its agreement
with a deployment depends on the omitted costs and feedback described below.

## Offline KV import

When P execution does not depend on D load or completion, use
`SimulatorOfflineKVConnector` to compose independent `NON_BLOCKING` replays.
No running P server or broker is needed during the D phase. Configure the D
engine with `mode="NON_BLOCKING"` and this native KV connector configuration:

```json
{
  "kv_connector": "SimulatorOfflineKVConnector",
  "kv_connector_module_path": "vllm_simulator.plugin.kv_transfer.offline",
  "kv_role": "kv_consumer"
}
```

Declare the available prompt KV on each D request through the native engine API:

```python
params = SamplingParams(
    max_tokens=32, temperature=0, ignore_eos=True,
    extra_args={
        "simulation": {"created_time": prefill_finished_at, "total_requests": shard_size},
        "kv_transfer_params": {"offline_kv_tokens": len(prompt_token_ids)},
    },
)
```

`offline_kv_tokens` must be an integer from zero to the prompt length. Missing
or zero means normal prefill. A partial snapshot only skips its cached prefix;
a full snapshot still recomputes the final prompt token for sampling. Native
KV allocation, local prefix hits, batching, waiting, preemption and release
remain active. The snapshot is assumed to remain available for reload after
preemption; previously generated tokens are recomputed as needed. This
connector requires DP=PP=PCP=DCP=1 and one full-attention KV group; TP may differ
between the P and D phases. Offline import and online P/D handoff flags cannot
be combined, and external KV reads must remain enabled.

## Run the sweep

Run the example from the repository root:

```bash
VLLM_PLUGINS=simulator python tools/vllm-simulator/examples/offline_pd_ratio_sweep.py \
  --num-requests 1000 --input-len 128 --output-len 32 --request-rate 100 \
  --gpu-budget 4 --ratios 1:3 2:2 3:1 --output pd-ratios.json
```

The example sorts requests by original arrival time and splits them round-robin
across x logical P instances. Each P request generates one token; its virtual
`simulation.last_event_time` becomes the prefill completion boundary. It then
sorts those completion events and round-robins them across y logical D
instances, using each P completion time plus `--transfer-delay-ms` as the new D
`created_time`. Equal timestamps retain the original trace order.

All partitions retain the same virtual time origin. Logical instances are
replayed sequentially with a cleared KV pool between partitions, reusing one
native engine per phase. This evaluates x/y logical instances without launching
x/y concurrent servers. Executing P and D phases sequentially in wall time does
not serialize them in the resulting virtual timeline.

## Fixed datasets and resource budgets

Supply a fixed tokenized dataset with `--dataset trace.jsonl`. Each JSONL row
has this shape; IDs must be unique, output lengths positive, and times are
non-negative virtual seconds:

```json
{"request_id":"r0","prompt_token_ids":[42,43,44],"output_length":8,"created_time":0.0}
{"request_id":"r1","prompt_token_ids":[45,46],"output_length":16,"created_time":0.01}
```

For heterogeneous TP, compare allocations using the same GPU budget:

```bash
VLLM_PLUGINS=simulator python tools/vllm-simulator/examples/offline_pd_ratio_sweep.py \
  --dataset trace.jsonl --p-tp 1 --d-tp 2 \
  --gpu-budget 6 --ratios 4:1 2:2 --output pd-ratios.json
```

Without `--ratios`, all positive P:D pairs satisfying
`x * p_tp + y * d_tp == gpu_budget` are scanned. `--p-predictor` and
`--d-predictor` accept JSON predictor settings; calibrate them for the respective
hardware and TP. Default linear settings demonstrate the workflow and do not
predict TP scaling automatically.

## Throughput accounting

The table reports request/s and output-token/s. Both use the global duration
`max(D_finished_at) - min(original_created_time)`, including P execution,
transfer delay, D queueing and decode. Node throughputs are not added together.
The JSON output also records original arrival, P completion, D arrival and
final completion per request. P's single output token is discarded, and D
produces the full requested output length; tokens are counted once. As in the
online connector, recomputing the final prompt token on D adds boundary work.

## Approximate assumptions and limits

This is an approximation with fixed arrivals and deterministic round-robin
routing. KV snapshots and imports have zero modeled storage/copy cost. The
optional transfer delay is constant and independent per request; network
contention, TP redistribution cost, retained P KV, D-to-P backpressure, global
admission limits and live load-dependent routing are omitted. The best ratio
is specific to the dataset, arrival rate, resource budget and predictors;
light workloads may yield ties. A D admission limit that stalls P, for example, would change the P completion
trace and invalidate this one-way decomposition. Use
[`BLOCKING` with the shared broker](disaggregated-serving.md#independently-deployed-prefill-and-decode)
to validate candidates when transfer contention and feedback affect the result.
