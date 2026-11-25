"""Single-node DashLLM -> vLLM -> v6d smoke test, using BLOCKING + linear.

Run in the serving image with an existing local model (no downloads):
    python test/test_dashllm_single_node.py --model /path/to/model
    python test/test_dashllm_single_node.py --model /path/to/model --target-dp 2

The Python entrypoint is the serving worker used by `dashllm_cmd serving`.
A private Redis, ports, sockets, and logs isolate this run from other services.
Populate v6d, reset only GPU L1, then replay identical prompts to verify L2 loads.
"""

import argparse
import json
import os
import random
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections import namedtuple
from contextlib import ExitStack
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

Ports = namedtuple(
    "Ports",
    "grpc control worker_control redis v6d v6d_rpc sched_rpc vl_v6d vl_v6d_rpc dp_master dp_rpc",
)
MAX_MODEL_LEN = 16384


def stop(process):
    if process is None:
        return
    try:
        # The leader can exit before its EngineCore children; still stop its group.
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def list_v6d_objects(base_url):
    request = Request(
        f"{base_url}/objects/list",
        data=b'{"tier":"vineyard"}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--http-benchmark",
        action="store_true",
        help="Validate token-trace replay through the HTTP router and native vLLM benchmark",
    )
    parser.add_argument("--target-tp", type=int)
    parser.add_argument(
        "--target-dp",
        type=int,
        choices=(1, 2),
        default=1,
        help="With DP2, verify the v6d EngineCore interval on both ranks and send a request",
    )
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--mamba-cache-dtype")
    parser.add_argument("--block-align-size", type=int)
    parser.add_argument("--speculative-tokens", type=int)
    parser.add_argument("--speculative-method", default="qwen4_exp_mtp")
    parser.add_argument("--csa-cp-local", action="store_true")
    parser.add_argument("--csa-indexer-backend", choices=("triton", "pai_deep_gemm"))
    parser.add_argument("--expected-block-size", type=int)
    parser.add_argument("--expected-spec-sizes", type=int, nargs="+")
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Truncate decoder layers via hf_overrides to shorten large-model startup",
    )
    parser.add_argument(
        "--remote-ple",
        action="store_true",
        help="Serve a PLE model in remote_v6d mode; HiSim models admission",
    )
    parser.add_argument(
        "--ple-admission-ms",
        type=float,
        default=500.0,
        help="Constant simulated PLE admission per request with --remote-ple",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=3,
        help="Long requests per populate/replay phase",
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="Startup timeout in seconds"
    )
    args = parser.parse_args()
    args.model = args.model.resolve()
    if not (args.model / "config.json").is_file():
        parser.error("--model must contain a local config.json")
    if args.requests < 1 or args.timeout < 1:
        parser.error("requests and timeout must be positive")
    if args.ple_admission_ms <= 0:
        parser.error("ple-admission-ms must be positive")
    if args.target_tp is not None and args.target_tp < 1:
        parser.error("target-tp must be positive")
    if args.speculative_tokens is not None and args.speculative_tokens < 0:
        parser.error("speculative-tokens must be nonnegative")
    if args.num_layers is not None and args.num_layers < 1:
        parser.error("num-layers must be positive")
    for binary in ("redis-server", "dashservingd", "v6d"):
        if shutil.which(binary) is None:
            parser.error(f"Required serving-image binary is missing: {binary}")
    return args


def reserve_ports():
    """Reserve distinct local ports, then release them immediately before launch."""
    with ExitStack() as reserved:
        ports = []
        for _ in Ports._fields:
            listener = reserved.enter_context(socket.socket())
            listener.bind(("127.0.0.1", 0))
            ports.append(listener.getsockname()[1])
    return Ports(*ports)


def write_platform_config(work):
    config = {
        "platform": {
            "accelerator": {"name": "h100_sxm", "hbm_capacity_gb": 80},
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "num_device_per_node": 1,
        },
        "predictor": {"name": "linear", "base_ms": 2.0, "per_token_ms": 0.01},
    }
    (work / "config.json").write_text(json.dumps(config))


def write_sitecustomize(work):
    # Bootstrap every Python child, clear DashServing's GPU assignment for CPU
    # simulation, and disable model staging using a private, absent directory.
    (work / "sitecustomize.py").write_text(
        "import os\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
        "import hisim._pth_bootstrap\n"
        "from hisim.hook import BaseHook, install_module_hooks\n"
        "class ModelStageHook(BaseHook):\n"
        "    HOOK_MODULE_NAME = 'dashllm.core.backend.model_integrity_check'\n"
        "    @classmethod\n"
        "    def hook(cls, module):\n"
        "        module.model_path_in_shm = os.path.join(os.environ['HISIM_OUTPUT_DIR'], 'no-staging', 'model')\n"
        "install_module_hooks([ModelStageHook])\n"
    )


def create_workspace(root):
    (root / "tmp").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dashllm-smoke-", dir=root / "tmp"))
    (work / "v6d").mkdir()
    (work / "logs").mkdir()
    (work / "no-staging").mkdir()
    write_platform_config(work)
    write_sitecustomize(work)
    return work


def engine_config(ports):
    """vLLM engine arguments: CPU-simulated dummy weights, prefix caching, and v6d KV offload."""
    return {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "load_format": "dummy",
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "mamba_cache_mode": "light",
        "async_scheduling": False,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": 8192,
        "block_size": 128,
        "num_gpu_blocks_override": 128,
        "kv_transfer_config": {
            "kv_connector": "HybridConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "backend": "v6d_object",
                "rpc_port": ports.sched_rpc,
            },
        },
    }


def v6d_args(work, vineyard_size="8G"):
    """v6d daemon flags: tiered peer identified by this run, sparse arena of the given size."""
    return (
        f"--peer=tiered_vineyard --peer-id {work.name} --vineyard-size={vineyard_size}"
    )


def layer_overrides(model, num_layers):
    """hf_overrides keeping the first num_layers decoder layers and their types."""
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config")
    target = config if text is None else text
    total = target["num_hidden_layers"]
    if num_layers > total:
        raise ValueError(f"num-layers {num_layers} exceeds the model's {total}")
    overrides = {"num_hidden_layers": num_layers}
    if "layer_types" in target:
        overrides["layer_types"] = target["layer_types"][:num_layers]
    return overrides if text is None else {"text_config": overrides}


def build_environment(root, work, model, ports):
    env = os.environ.copy()
    for key in list(env):
        # RANK_* and DS_UNIT_NODE_NUM describe a multi-node deployment.
        if key.startswith(
            ("HISIM_", "VLLM_", "DS_LLM_", "DS_MODEL_AGENT", "SCR_", "RANK_")
        ):
            env.pop(key)
    for key in (
        "MASTER_ADDRESS",
        "MASTER_PORT",
        "DASH_GEN_ARGS",
        "DS_AUTH_TOKEN",
        "DS_UNIT_NODE_NUM",
        "_DS_DEPENDENCY_WHEELS",
    ):
        env.pop(key, None)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(work), str(root / "src"), env.get("PYTHONPATH", "")]
            ),
            "HISIM_ENABLE": "1",
            "HISIM_OUTPUT_MODE": "BLOCKING",
            # BLOCKING admission hit snapshots are opt-in.
            "HISIM_RECORD_STATS": "1",
            "HISIM_MAX_TOKENS": "1",
            "HISIM_CONFIG_PATH": str(work / "config.json"),
            "HISIM_OUTPUT_DIR": str(work),
            "MASTER_ADDRESS": "127.0.0.1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONHASHSEED": "42",
            "SRPC_STREAM_DISABLE_RDMA": "1",
            "SRPC_THREAD_NUM": "1",
            "VLLM_ENABLE_DASHLOG": "0",
            "VLLM_SERVER_DEV_MODE": "1",
            "DS_ENABLE_DSERV": "1",
            "DS_LLM_ENGINE": "vllm",
            "model_path": str(model),
            "DS_LLM_MULTI_ENGINE_NUM": "1",
            "DS_LLM_IGNORE_WARMUP": "1",
            # Expose L1 invalidation only in these private workers, without SCR lifecycle.
            "SCR_NEXT_CONTAINER": "1",
            "SCR_ENABLE": "0",
            "DS_LLM_ENABLE_SCR_CROSS_DEPLOYMENT_RESTORE": "0",
            "VLLM_V6D_ASYNC_REGISTER": "0",
            "DASHGEN_DEPLOYMENT_ROLE": "",
            "DSV_CONTROL_SERVER_BASE_PORT": str(ports.worker_control),
            "DS_LOG_ROOT_DIR": str(work / "logs"),
            "DS_MODEL_AGENT": "0",
            "DS_SCR_CKPT_MGR_ENABLE": "0",
            "DS_MODEL_PRELOAD_TO_SHM": "0",
            "DS_MODEL_INTEGRITY_CHECK": "0",
            "DS_METER_DISABLE": "1",
            "OTEL_SDK_DISABLED": "true",
            "DS_LLM_LAUNCH_V6D": "1",
            "DS_LLM_SHARE_V6D": "1",
            "DS_LLM_PROC_RANK": "0",
            "DS_LLM_V6D_SOCKET_DIR": str(work / "v6d"),
            "V6D_PORT": str(ports.v6d),
            "V6D_RPC_PORT": str(ports.v6d_rpc),
            # VL models start a DataRouter v6d whose default socket is shared in
            # /dev/shm; a stale one from an earlier run would be "reused" forever.
            "DS_VL_V6D_SOCKET": str(work / "v6d" / "vineyard.vl.sock"),
            "DS_VL_V6D_PORT": str(ports.vl_v6d),
            "DS_VL_V6D_RPC_PORT": str(ports.vl_v6d_rpc),
            "V6D_ENABLE_TRACKER": "1",
            "V6D_KVCM_MODE": "disabled",
            "KVS_METASERVICE_REDIS_URI": f"redis://127.0.0.1:{ports.redis}",
            "SPECTRUM_DEPLOYMENT_NAME": work.name,
            "V6D_ARGS": v6d_args(work),
            "ENABLE_LOCAL_SSD": "0",
            "ENABLE_SHAREDFS": "0",
            "ENABLE_KVCS": "0",
            "DS_LLM_ENGINE_CONFIG": json.dumps(engine_config(ports)),
        }
    )
    return env


def worker_command(work, ports, engine_count):
    return [
        sys.executable,
        "-m",
        "dashserving.entrypoints.cli",
        "run",
        "dashllm.worker.llm:worker",
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.grpc),
        "--control-port",
        str(ports.control),
        "--gpus",
        str(engine_count),
        "--object-store",
        str(work / "objects"),
        "--log-dir",
        str(work / "logs"),
    ]


def start_services(processes, work, ports, env, log, engine_count):
    processes.append(
        subprocess.Popen(
            [
                "redis-server",
                "--bind",
                "127.0.0.1",
                "--port",
                str(ports.redis),
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(work),
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    )
    processes.append(
        subprocess.Popen(
            worker_command(work, ports, engine_count),
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    )


def wait_until_healthy(processes, port, timeout, log_path):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError(f"A service exited; inspect {log_path}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Service startup timed out; inspect {log_path}")


def max_prompt_length(block_size):
    """Worst-case prompt length the request loop can generate for this block size."""
    return 2 * block_size + max(1, block_size // 4)


def cache_block_size(v6d_url):
    with urlopen(f"{v6d_url}/kv-cache/spec", timeout=5) as response:
        block_size = int(json.load(response)["block_size"])
    assert block_size > 0 and max_prompt_length(block_size) < MAX_MODEL_LEN, (
        f"Cache block size exceeds this test's context budget: {block_size}"
    )
    return block_size


def snapshot_objects(v6d_url, destination):
    objects = list_v6d_objects(v6d_url)
    destination.write_text(json.dumps(objects, indent=2))
    return objects


def generate_once(client, prompt, log_path, request_id):
    tokens, finished = [], False
    for response in client.generate(
        request_id=request_id,
        prompt={"prompt_token_ids": prompt},
        params={"max_tokens": 1, "ignore_eos": True},
        capability={"type": "generate"},
    ):
        if response.is_error_response():
            raise RuntimeError(
                f"{response.error_msg or 'Generation failed'}; inspect {log_path}"
            )
        if response.token_ids:
            tokens.extend(response.token_ids[0])
        finished |= response.is_finished()
    assert finished and len(tokens) == 1, (finished, len(tokens))
    return tokens


def make_prompts(request_count, block_size):
    rng = random.Random(42)
    prompts = []
    for _ in range(request_count):
        # Schedule two complete physical blocks, including large hybrid blocks.
        prompt_length = 2 * block_size + rng.randint(1, max(1, block_size // 4))
        prompts.append([rng.randint(100, 1000) for _ in range(prompt_length)])
    return prompts


def wait_for_request_hit(log_path, offset, request_id, timeout=5):
    """Read the completed request's admission snapshot, not stale hit counters.

    DashLLM suffixes the engine request ID, and some builds hand vLLM one
    prompt token fewer, so match the client ID prefix rather than the length.
    """
    pattern = re.compile(
        r"\[HitRate\] rid=(\S+) input_len=(\d+) local_hit=(-?\d+) ext_hit=(\d+)"
    )
    deadline = time.monotonic() + timeout
    while True:
        with log_path.open("rb") as log:
            log.seek(offset)
            text = log.read().decode(errors="replace")
        matches = [
            match
            for match in pattern.finditer(text)
            if match[1] == request_id or match[1].startswith(f"{request_id}-")
        ]
        if len(matches) == 1:
            match = matches[0]
            return {
                "engine_request_id": match[1],
                "local_hit_tokens": int(match[3]),
                "external_hit_tokens": int(match[4]),
            }
        if len(matches) > 1:
            raise AssertionError(f"Ambiguous request hit snapshots; inspect {log_path}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"Missing request hit snapshot; inspect {log_path}")
        time.sleep(0.1)


def run_requests(
    client, prompts, log_path, phase, require_external=False, min_elapsed=0.0
):
    rows = []
    for index, prompt in enumerate(prompts):
        request_id = f"smoke-{uuid.uuid4().hex}"
        offset = log_path.stat().st_size
        started = time.monotonic()
        tokens = generate_once(client, prompt, log_path, request_id)
        elapsed = time.monotonic() - started
        hit = wait_for_request_hit(log_path, offset, request_id)
        if require_external and (
            hit["external_hit_tokens"] <= 0 or hit["local_hit_tokens"] != 0
        ):
            raise AssertionError(f"Replay did not load L2 after L1 reset: {hit}")
        if elapsed < min_elapsed:
            raise AssertionError(
                f"{phase} request finished in {elapsed:.3f}s, before the "
                f"{min_elapsed:.3f}s simulated PLE admission"
            )
        rows.append(
            {
                "phase": phase,
                "input_length": len(prompt),
                "output_length": len(tokens),
                "elapsed_s": elapsed,
                **hit,
            }
        )
        print(
            f"{phase} request {index + 1}: input={len(prompt)}, output={len(tokens)}, "
            f"local_hit={hit['local_hit_tokens']}, ext_hit={hit['external_hit_tokens']}, "
            f"elapsed={elapsed:.3f}s",
            flush=True,
        )
    return rows


def reset_l1(port):
    """Use the worker control callback; never reset the v6d connector."""
    request = Request(
        f"http://127.0.0.1:{port}/invalidate_kv_cache",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        status, result = response.status, json.load(response)
    if status != 200 or result.get("status") != "kv_cache_invalidated":
        raise AssertionError(f"L1 cache reset failed: HTTP {status}, {result}")


def assert_l2_retained(before, after):
    sizes = {obj["key"]: obj["size"] for obj in after["objects"]}
    missing = [
        obj["key"] for obj in before["objects"] if sizes.get(obj["key"]) != obj["size"]
    ]
    if missing:
        raise AssertionError(f"L1 reset removed or changed L2 objects: {missing}")


def wait_for_stored_objects(v6d_url, before, timeout=30):
    """Inspect only daemon-owned metadata; never create objects here."""
    before_keys = {obj["key"] for obj in before["objects"]}
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        after = list_v6d_objects(v6d_url)
        new_objects = [
            obj
            for obj in after["objects"]
            if obj["key"] not in before_keys and obj["size"] > 0
        ]
        signature = {(obj["key"], obj["size"]) for obj in new_objects}
        # Require all visible stores to release leases and settle across polls.
        if new_objects and all(not obj["leased"] for obj in new_objects):
            if signature == previous:
                return after, new_objects
            previous = signature
        else:
            previous = None
        if time.monotonic() >= deadline:
            raise AssertionError(f"No completed KV cache stores in v6d: {after}")
        time.sleep(0.5)


def run_http_benchmark(root, work, model, ports, processes, log):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        http_port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(HISIM_ENABLE="0", SIM_COLLECTOR_ENABLE="0", PYTHONPATH=str(root / "src"))
    router = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hisim.benchmark.router",
            "--port",
            str(http_port),
            "--dashllm-address",
            f"127.0.0.1:{ports.grpc}",
        ],
        env=env,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    processes.append(router)
    wait_until_healthy([router], http_port, 30, work / "service.log")
    trace = work / "http-trace.jsonl"
    lengths = [37, 89, 151]
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "input_ids": [100 + i] * length,
                    "input_length": length,
                    "output_length": 1,
                    "timestamp": 1000 + i * 0.2,
                }
            )
            for i, length in enumerate(lengths)
        )
    )
    for scale in (0.5, 2):
        destination = work / f"http-benchmark-{scale}.json"
        manifest = work / f"http-trace-{scale}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hisim.benchmark.serve",
                "bench",
                "serve",
                "--backend",
                "openai",
                "--model",
                str(model),
                "--base-url",
                f"http://127.0.0.1:{http_port}",
                "--dataset-name",
                "hisim",
                "--dataset-path",
                str(trace),
                "--num-prompts",
                "3",
                "--timestamp-scale",
                str(scale),
                "--ignore-eos",
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(work),
                "--result-filename",
                destination.name,
                "--trace-manifest",
                str(manifest),
            ],
            env=env,
            stdout=log,
            stderr=log,
            check=True,
            timeout=120,
        )
        result = json.loads(destination.read_text())
        schedule = json.loads(manifest.read_text())
        assert result["completed"] == 3 and result["failed"] == 0, result
        assert result["input_lens"] == lengths, result
        assert result["total_output_tokens"] == 3, result
        assert result["mean_ttft_ms"] > 0, result
        assert abs(schedule["arrival_span_seconds"] - 0.4 * scale) < 1e-9, schedule
        assert schedule["max_concurrency"] is None, schedule
    print(
        "HTTP benchmark: original token lengths, scaled timestamps, native TTFT and output counts verified",
        flush=True,
    )


def main():
    args = parse_args()

    try:
        from dashserving.client.llm_client import LLMClient
    except ImportError:  # Older DashServing releases
        from dashserving.client._rust import LLMClient
    from dashserving.config import DashClientConfig

    ports = reserve_ports()
    root = Path(__file__).resolve().parents[1]
    work = create_workspace(root)
    env = build_environment(root, work, args.model, ports)
    engine = json.loads(env["DS_LLM_ENGINE_CONFIG"])
    for field, value in (
        ("tensor_parallel_size", args.target_tp),
        ("kv_cache_dtype", args.kv_cache_dtype),
        ("mamba_cache_dtype", args.mamba_cache_dtype),
        ("block_align_size", args.block_align_size),
    ):
        if value is not None:
            engine[field] = value
    if args.speculative_tokens is not None:
        engine["speculative_config"] = (
            {
                "method": args.speculative_method,
                "num_speculative_tokens": args.speculative_tokens,
            }
            if args.speculative_tokens
            else None
        )
    if args.num_layers is not None:
        engine["hf_overrides"] = layer_overrides(args.model, args.num_layers)
    if args.target_dp > 1:
        engine.update(
            data_parallel_size=args.target_dp, data_parallel_rpc_port=ports.dp_rpc
        )
        env.update(
            DS_LLM_MULTI_ENGINE_NUM=str(args.target_dp),
            VLLM_DP_MASTER_PORT=str(ports.dp_master),
        )
    env["DS_LLM_ENGINE_CONFIG"] = json.dumps(engine)
    simulator = json.loads((work / "config.json").read_text())
    min_elapsed = 0.0
    if args.remote_ple:
        # Non-PD serving; HiSim restores the v6d+KVT connector remote PLE needs.
        env.update(VLLM_PLE_OFFLOAD_BACKEND="remote_v6d", VLLM_PLE_CPU_OFFLOAD="1")
        # A constant, large admission keeps the delay visible in request latency.
        simulator["ple"] = {
            "admission": {"base_ms": args.ple_admission_ms, "per_1k_tokens_ms": 0}
        }
        min_elapsed = args.ple_admission_ms / 1000
    env["HISIM_CONFIG_JSON"] = json.dumps(simulator)
    if args.csa_cp_local:
        env.update(
            VLLM_GATED_RESIDUAL_TPSP="1",
            VLLM_QSA_PREFILL_ONLY_INDEXER_USE_CP="1",
            VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD="0",
        )
    if args.csa_indexer_backend is not None:
        env["VLLM_CSA_INDEXER_BACKEND"] = args.csa_indexer_backend
    print(f"Artifacts: {work}", flush=True)
    processes = []
    log_path = work / "service.log"
    with log_path.open("w") as log:
        try:
            start_services(processes, work, ports, env, log, args.target_dp)
            wait_until_healthy(processes, ports.worker_control, args.timeout, log_path)
            # Worker health can precede the serving port bind.
            deadline = time.monotonic() + 30
            while True:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError(
                        f"Service exited before serving bind: {log_path}"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", ports.grpc), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Serving port did not bind: {log_path}"
                        ) from None
                    time.sleep(0.2)
            assert stat.S_ISSOCK((work / "v6d/vineyard.sock").stat().st_mode), (
                "DashLLM did not launch v6d"
            )
            v6d_url = f"http://127.0.0.1:{ports.v6d}"
            block_size = cache_block_size(v6d_url)
            with urlopen(f"{v6d_url}/kv-cache/spec", timeout=5) as response:
                layout = json.load(response)
            (work / "kv-spec.json").write_text(json.dumps(layout, indent=2))
            spec_sizes = [int(group["object_size_bytes"]) for group in layout["groups"]]
            if args.expected_block_size is not None:
                assert block_size == args.expected_block_size, (
                    block_size,
                    args.expected_block_size,
                )
            if args.expected_spec_sizes is not None:
                assert spec_sizes == args.expected_spec_sizes, (
                    spec_sizes,
                    args.expected_spec_sizes,
                )
            print(
                f"KV layout: block_size={block_size}, spec_sizes={spec_sizes}",
                flush=True,
            )
            client = LLMClient(
                f"127.0.0.1:{ports.grpc}",
                capability={"type": "generate"},
                config=DashClientConfig(request_timeout_ms=60000),
            )
            marker = "EngineCore thread switch interval set to 0.25 ms"
            expected = args.target_dp if args.target_dp > 1 else 0
            count = log_path.read_text(errors="replace").count(marker)
            assert count == expected, (count, expected, log_path)
            if args.target_dp > 1:
                generate_once(
                    client,
                    [100] * (2 * block_size),
                    log_path,
                    f"smoke-{uuid.uuid4().hex}",
                )
                if args.http_benchmark:
                    run_http_benchmark(root, work, args.model, ports, processes, log)
                print(
                    "PASS: DP2 DashLLM/vLLM/v6d EngineCore switch interval and inference"
                )
                return
            before = snapshot_objects(v6d_url, work / "v6d-before.json")
            print(
                f"v6d: {v6d_url}, block_size={block_size} tokens, objects_before={before['total']}",
                flush=True,
            )
            short_prompt = [100] * max(1, min(64, block_size - 1))
            rows = run_requests(
                client, [short_prompt], log_path, "short", min_elapsed=min_elapsed
            )
            prompts = make_prompts(args.requests, block_size)
            rows.extend(
                run_requests(
                    client, prompts, log_path, "populate", min_elapsed=min_elapsed
                )
            )
            after, stored = wait_for_stored_objects(v6d_url, before)
            (work / "v6d-after.json").write_text(json.dumps(after, indent=2))
            declared_sizes = {
                str(g["group_id"]): int(g["object_size_bytes"])
                for g in layout["groups"]
            }
            for obj in stored:
                group_id = obj["key"].rsplit("_", 1)[1]
                assert obj["size"] == declared_sizes[group_id], obj
            declared_bytes = sum(obj["size"] for obj in stored)
            print(
                f"v6d: objects_after={after['total']}, new_released_objects={len(stored)}, "
                f"declared_cache_bytes={declared_bytes}",
                flush=True,
            )
            reset_l1(ports.worker_control)
            retained = snapshot_objects(v6d_url, work / "v6d-after-l1-reset.json")
            assert_l2_retained(after, retained)
            print("L1 reset succeeded; v6d L2 objects retained", flush=True)
            rows.extend(
                run_requests(
                    client,
                    prompts,
                    log_path,
                    "replay",
                    require_external=True,
                    min_elapsed=min_elapsed,
                )
            )
            (work / "cache-replay.json").write_text(json.dumps(rows, indent=2))
            snapshot_objects(v6d_url, work / "v6d-after-replay.json")
            if args.remote_ple:
                text = log_path.read_text(errors="replace")
                for marker in (
                    "remote PLE uses HybridConnector backend v6d_object+kvt",
                    "[PLE Hijack] PLEConnector uses simulated admission",
                ):
                    assert marker in text, (
                        f"Missing remote PLE marker {marker!r}; inspect {log_path}"
                    )
                print(
                    "Remote PLE: every request waited for simulated admission",
                    flush=True,
                )
            if args.http_benchmark:
                run_http_benchmark(root, work, args.model, ports, processes, log)
            print(
                "PASS: single-node DashLLM/vLLM/v6d, BLOCKING + linear, "
                "native KV stores and L2 loads after L1 reset",
                flush=True,
            )
        finally:
            # EngineCore can detach from the daemon's process group during shutdown.
            import psutil

            descendants = []
            for process in processes:
                try:
                    descendants.extend(
                        psutil.Process(process.pid).children(recursive=True)
                    )
                except psutil.NoSuchProcess:
                    pass
            for process in reversed(processes):
                stop(process)
            _, alive = psutil.wait_procs(descendants, timeout=5)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)


if __name__ == "__main__":
    main()
