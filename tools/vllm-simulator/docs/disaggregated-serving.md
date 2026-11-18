# Disaggregated serving

## Independently deployed prefill and decode

Existing deployment files using built-in `NixlConnector` or
`MooncakeConnector` can retain their native `kv-transfer-config`, including
`kv_buffer_device: cuda`, engine ID and KV-load failure policy. Add only the
simulation transport settings:

```bash
VLLM_PLUGINS=simulator vllm serve --config prefill.yaml \
  --additional-config '{"simulator":{"mode":"BLOCKING","kv_transfer":{"broker_url":"http://127.0.0.1:9000","kv_role":"kv_producer"}}}'

VLLM_PLUGINS=simulator vllm serve --config decode.yaml \
  --additional-config '{"simulator":{"mode":"BLOCKING","kv_transfer":{"broker_url":"http://127.0.0.1:9000","kv_role":"kv_consumer"}}}'
```

For example, a shared native YAML section can remain:

```yaml
kv-transfer-config:
  kv_connector: NixlConnector
  kv_role: kv_both
  kv_buffer_device: cuda
  kv_connector_extra_config:
    backends: [UCX]
```

The plugin substitutes `SimulatorPDConnector` before connector initialization;
neither native transport library nor GPU buffers are needed. `kv_both` requires
an explicit simulator `kv_role`; a native producer/consumer role is inherited
and cannot be contradicted. `broker_url` is required; `rpc_timeout_s` and
`poll_interval_s` are optional. These settings also work with an explicit
`SimulatorPDConnector`; its existing extra-config syntax remains supported.

The supported transport-only extras are NIXL `backends: [UCX]` and
`ucx_backend_params`, or Mooncake `mooncake_protocol`, `device_name` and
`bootstrap_server_address`.
They are retained in the original configuration but are not executed: the
broker's bandwidth/latency/copy settings determine simulated transfer timing.
Native bootstrap/side-channel addresses identify native protocol services;
they are not aliases for the shared simulator broker. They remain in the
target configuration and are not contacted. `broker_url` must identify a
separately started simulator broker, even when a native bootstrap address is
present. Automatic broker discovery or embedded broker startup is not provided.
Other native extras, custom connector modules, GDS and MultiConnector are not
automatically adapted. Unsupported settings are rejected rather than silently
changing their semantics. Existing P/D topology and layout restrictions below
still apply; this adapter does not implement native NIXL/Mooncake wire protocols
or replace the router's handoff logic.

Predictor factories receive the preserved native connector configuration and
target topology, including the target's local DP count. Dynamic DP rank/index
fields reflect the executing engine.

`SimulatorPDConnector` connects separately served P and D instances through a
shared metadata broker. Each instance keeps its native scheduler and KV pool.
The broker models a shared FIFO network and per-D copy queue using its own
monotonic clock. It transfers metadata and completion events, not KV tensors.

Start the broker in another terminal:

```bash
.venv/bin/python -m vllm_simulator.plugin.kv_transfer.broker \
  --host 0.0.0.0 --port 9000 \
  --bandwidth-bytes-per-second 25000000000 --latency-ms 0.05 \
  --reshard-bandwidth-bytes-per-second 1000000000000 --lease-seconds 60
```

Start P and D as ordinary independent servers. These examples run on one host;
for separate hosts, replace `broker_url` with the broker's reachable address
and route clients to the corresponding P/D addresses.

```bash
VLLM_PLUGINS=simulator vllm serve Qwen/Qwen3-0.6B \
  --load-format dummy --host 0.0.0.0 --port 8100 \
  --max-model-len 512 --block-size 16 --num-gpu-blocks-override 64 \
  --tensor-parallel-size 1 --enable-server-load-tracking \
  --additional-config '{"simulator":{"mode":"BLOCKING","predictor":{"name":"linear","base_ms":5}}}' \
  --kv-transfer-config '{"kv_connector":"SimulatorPDConnector","kv_connector_module_path":"vllm_simulator.plugin.kv_transfer.disaggregated","kv_role":"kv_producer","kv_connector_extra_config":{"broker_url":"http://127.0.0.1:9000"}}'

VLLM_PLUGINS=simulator vllm serve Qwen/Qwen3-0.6B \
  --load-format dummy --host 0.0.0.0 --port 8200 \
  --max-model-len 512 --block-size 16 --num-gpu-blocks-override 64 \
  --tensor-parallel-size 2 --enable-server-load-tracking \
  --additional-config '{"simulator":{"mode":"BLOCKING","predictor":{"name":"linear","base_ms":5}}}' \
  --kv-transfer-config '{"kv_connector":"SimulatorPDConnector","kv_connector_module_path":"vllm_simulator.plugin.kv_transfer.disaggregated","kv_role":"kv_consumer","kv_connector_extra_config":{"broker_url":"http://127.0.0.1:9000"}}'
```

Your router uses the native `kv_transfer_params` handoff convention:

1. Select P and submit `max_tokens=1` with
   `kv_transfer_params={"do_remote_decode": true}`.
2. Read P's response `kv_transfer_params`. P's compute has finished, but its
   blocks remain pinned until transmission completes or the offer expires.
3. Select D and submit the original prompt and desired output length with P's
   handoff parameters. Forward the whole dict when possible. The native
   `remote_engine_id` and `remote_request_id` source descriptor also resolves
   the handoff; a simulator transfer ID is not required from the router.
4. D reserves real native receive blocks, enters `WAITING_FOR_REMOTE_KVS`, and
   waits for the broker's transfer and repack completion. Native
   `finished_recving` resumes D; `finished_sending` releases P's blocks.

For example, ordinary HTTP clients can submit the two stages:

```python
import time
import requests

model = "Qwen/Qwen3-0.6B"
request = {"model": model, "prompt": [42, 43, 44],
           "temperature": 0, "ignore_eos": True}
started = time.monotonic()
p = requests.post(
    "http://localhost:8100/v1/completions",
    json={**request, "max_tokens": 1,
          "kv_transfer_params": {"do_remote_decode": True}}, timeout=30,
)
p.raise_for_status()
d = requests.post(
    "http://localhost:8200/v1/completions",
    json={**request, "max_tokens": 16,
          "kv_transfer_params": p.json()["kv_transfer_params"]}, timeout=30,
)
d.raise_for_status()
print(d.json(), "end_to_end_s", time.monotonic() - started)
```

The same handoff works with `/v1/chat/completions` and streaming. The D prompt
must tokenize identically to the P prompt. P's single output token is discarded;
D recomputes the final prompt token for sampling and generates the user output.
P returns its response before waiting for transmission; waiting for a D receive
before returning the P response would deadlock a sequential P-then-D router.

P/D selection remains in the existing scheduler. Launch more P/D endpoints to
compare allocation ratios and keep the same workload, client concurrency and
arrival process. With P TP=1 and D TP=2, 4P1D and 2P2D each consume six simulated
GPUs. Measure end-to-end latency at the router/client and examine each native
endpoint's load and KV metrics to identify the saturated phase. Simulator
multi-instance runners and their built-in routing/sweep entry points have been
removed.

### Transfer model and lifecycle

Every P/D pair sharing the broker competes for its FIFO link. Transfer service
time is `latency + bytes / bandwidth`. Identical TP/packing transfers physical
blocks including padding. Different TP/packing sends the required KV-head
slices, selects one source replica and creates every required D replica.
Native padded KV capacity remains independent on each instance; local D prefix
hits reduce the transferred range. MLA latent state is one replicated KV head.

Optional destination repacking reads the payload and writes the destination
physical blocks at the configured aggregate per-D copy bandwidth. Omitting
`--reshard-bandwidth-bytes-per-second` models zero extra copy cost. The network,
compute and different destinations' copies can overlap. Individual NIC limits,
copy/compute memory-bandwidth interference and layer streaming are outside
this model.

Broker events are acknowledged and redelivered until acknowledged. Cancellation
and `--lease-seconds` expiry release orphan source offers and failed receive
reservations. Offers are single-consumer; mismatched, expired and reused
handoffs follow the native KV-load failure policy (`fail` by default).
After a broker restart the engines release old source blocks and apply the same
load failure policy to old receives. New handoffs use the new broker's event stream.
Once transmission completes, D can finish repacking even if P shuts down.
If publishing a P handoff fails, that request returns an error and releases
its KV blocks; the P engine remains available for subsequent requests.

D responses additionally expose `kv_transfer_params.simulation_kv_transfer`:
wire bytes, rank/head slices, source/destination physical bytes, destination
allocation wait, network queue/service time and repack queue/service time.
`kv_destination_wait_s` measures D's local time from its first KV match attempt
until receive blocks are allocated. `kv_offer_to_claim_s` measures the broker's
time from publication to claim, including routing and D admission. The broker
also measures network and repack durations. Do not subtract absolute P/D engine
`simulation` timestamps across hosts; use the router/client's monotonic clock
for end-to-end results.

Connector extra configuration accepts `broker_url` (required), `rpc_timeout_s`
(default 2 seconds) and `poll_interval_s` (default 0.005 seconds). Native engine
steps collect completions; empty steps long-poll briefly to avoid busy spinning.
The broker provides `/health` and `/stats` for bridge diagnostics.

Online P/D handoff requires `BLOCKING`, DP=PP=PCP=DCP=1 per endpoint, the same model
and model dtype, matching block token sizes and one full-attention KV group.
TP, scheduler limits, predictor and simulated memory can differ. Heterogeneous
TP requires compatible per-head content. Hybrid/sparse groups, quantization
conversion and PP/context-parallel redistribution are unsupported. This
connector models transfer behavior without loading NIXL/Mooncake or allocating
GPU buffers.

## Native serving and load observation

### Reuse a deployment configuration

Keep the same native vLLM YAML file for real and simulated serving. For example,
`deployment.yaml` can describe a two-node TP/PP deployment:

```yaml
model: Qwen/Qwen3-0.6B
tensor-parallel-size: 2
pipeline-parallel-size: 2
nnodes: 2
node-rank: 0
master-addr: 192.0.2.1
distributed-executor-backend: mp
max-model-len: 512
```

```bash
# Real deployment: run the usual head/worker commands on the target machines.
vllm serve --config deployment.yaml

# Simulation: run this once on the simulation host.
VLLM_PLUGINS=simulator vllm serve --config deployment.yaml \
  --additional-config '{"simulator":{"mode":"BLOCKING","predictor":{"name":"linear","base_ms":5}}}'
```

The plugin preserves the requested topology for model layout, capacity and
prediction, while setting execution TP/PP/PCP/DCP, world size and node count to
one, node rank to zero and the model-worker master address to loopback. Model
weights are never loaded; `--load-format dummy` and
`--num-gpu-blocks-override` are optional. Without a block override, capacity is
derived from the simulated device, model storage and native memory settings.
Cache planning retains the native attention backends' layout preferences,
including block-outermost layouts for hybrid models with mixed sparse cache pages.

For internal DP load balancing, all DP engines run locally and use a loopback
DP master. For external DP load balancing, launch one simulated endpoint per
native DP rank; its DP rank, local DP count and DP communication address remain
unchanged. External DP requires `BLOCKING` and a MoE model. The DP backend must
be `mp`; Ray DP and hybrid load balancing are unsupported. A native Ray
model-worker backend is replaced by the simulator executor; its placement is
not reproduced.

This reuses the deployment configuration, not the complete multi-node launch
script: do not start the TP/PP worker-node commands or use `--headless` for
them. Each model-parallel instance has one simulator launcher. Internal DP
uses a single launcher for the whole group; external DP retains one per rank.
Native argument validation still applies before the plugin runs. The linear
predictor is only an example: topology-aware timing requires a calibrated
predictor, and physical GPU collectives and PP overlap are not executed.

### Submit requests

Start one endpoint per independently routed instance:

```bash
VLLM_PLUGINS=simulator vllm serve Qwen/Qwen3-0.6B \
  --load-format dummy --host 0.0.0.0 --port 8000 \
  --max-model-len 512 --num-gpu-blocks-override 64 \
  --enable-server-load-tracking \
  --additional-config '{"simulator":{"mode":"BLOCKING","predictor":{"name":"linear","base_ms":5}}}'
```

Use the existing vLLM interfaces for development and simulation deployment:

| Interface | Use |
| --- | --- |
| `/v1/completions`, `/v1/chat/completions` | Submit ordinary or streaming generation requests. |
| `/health`, `/v1/models` | Discover and check the instance. |
| `/load` | Read active HTTP request count; enable `--enable-server-load-tracking`. |
| `/metrics` | Read native running/waiting requests, KV usage and latency metrics. |

`/load` counts active HTTP requests, rather than available KV capacity. For
example, P may have returned its response while still retaining KV for D.
Use the native KV and waiting-request metrics alongside request counts. Metrics
follow vLLM's normal reporting cadence. No simulator-specific load endpoint or
router integration package is required.

Run your existing scheduler in front of the endpoints. Point the unmodified
serving benchmark at that scheduler or an individual instance:

```bash
vllm bench serve --backend openai --endpoint /v1/completions \
  --base-url http://localhost:8000 --model Qwen/Qwen3-0.6B \
  --dataset-name random --input-len 128 --output-len 32 \
  --num-prompts 100 --request-rate 10 --max-concurrency 8 --ignore-eos
```

Client timings include HTTP, routing, CPU scheduling and real transport-control
overhead as well as predicted GPU waits. Calibrate each instance's predictor
for its hardware, TP and workload. The default linear predictor is a functional
example and does not automatically accelerate larger TP sizes.
