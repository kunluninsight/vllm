"""Single-node DashLLM -> vLLM -> v6d smoke test, using BLOCKING + linear.

Run in the serving image with an existing local model (no downloads):
    python test/test_dashllm_single_node.py --model /path/to/model

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

Ports = namedtuple("Ports", "grpc control worker_control redis v6d v6d_rpc sched_rpc")
MAX_MODEL_LEN = 16384


def stop(process):
    if process is None or process.poll() is not None:
        return
    try:
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
        "scheduler": {"tp_size": 1, "ep_size": 1, "dp_size": 1},
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


def build_environment(root, work, model, ports):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("HISIM_", "VLLM_", "DS_LLM_", "DS_MODEL_AGENT", "SCR_")):
            env.pop(key)
    for key in ("MASTER_ADDRESS", "MASTER_PORT", "DASH_GEN_ARGS"):
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


def worker_command(work, ports):
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
        "1",
        "--object-store",
        str(work / "objects"),
        "--log-dir",
        str(work / "logs"),
    ]


def start_services(processes, work, ports, env, log):
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
            worker_command(work, ports),
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


def generate_once(client, prompt, log_path):
    tokens, finished = [], False
    for response in client.generate(
        request_id=f"smoke-{uuid.uuid4().hex}",
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


def wait_for_request_hit(log_path, offset, prompt_length, timeout=5):
    """Read the completed request's admission snapshot, not stale hit counters."""
    pattern = re.compile(
        r"\[HitRate\] rid=(\S+) input_len=(\d+) local_hit=(-?\d+) ext_hit=(\d+)"
    )
    deadline = time.monotonic() + timeout
    while True:
        with log_path.open("rb") as log:
            log.seek(offset)
            text = log.read().decode(errors="replace")
        matches = [
            match for match in pattern.finditer(text) if int(match[2]) == prompt_length
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


def run_requests(client, prompts, log_path, phase, require_external=False):
    rows = []
    for index, prompt in enumerate(prompts):
        offset = log_path.stat().st_size
        started = time.monotonic()
        tokens = generate_once(client, prompt, log_path)
        elapsed = time.monotonic() - started
        hit = wait_for_request_hit(log_path, offset, len(prompt))
        if require_external and (
            hit["external_hit_tokens"] <= 0 or hit["local_hit_tokens"] != 0
        ):
            raise AssertionError(f"Replay did not load L2 after L1 reset: {hit}")
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
    print(f"Artifacts: {work}", flush=True)
    processes = []
    log_path = work / "service.log"
    with log_path.open("w") as log:
        try:
            start_services(processes, work, ports, env, log)
            wait_until_healthy(processes, ports.worker_control, args.timeout, log_path)
            assert stat.S_ISSOCK((work / "v6d/vineyard.sock").stat().st_mode), (
                "DashLLM did not launch v6d"
            )
            v6d_url = f"http://127.0.0.1:{ports.v6d}"
            block_size = cache_block_size(v6d_url)
            before = snapshot_objects(v6d_url, work / "v6d-before.json")
            print(
                f"v6d: {v6d_url}, block_size={block_size} tokens, objects_before={before['total']}",
                flush=True,
            )
            client = LLMClient(
                f"127.0.0.1:{ports.grpc}",
                capability={"type": "generate"},
                config=DashClientConfig(request_timeout_ms=60000),
            )
            short_prompt = [100] * max(1, min(64, block_size - 1))
            rows = run_requests(client, [short_prompt], log_path, "short")
            prompts = make_prompts(args.requests, block_size)
            rows.extend(run_requests(client, prompts, log_path, "populate"))
            after, stored = wait_for_stored_objects(v6d_url, before)
            (work / "v6d-after.json").write_text(json.dumps(after, indent=2))
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
                )
            )
            (work / "cache-replay.json").write_text(json.dumps(rows, indent=2))
            snapshot_objects(v6d_url, work / "v6d-after-replay.json")
            print(
                "PASS: single-node DashLLM/vLLM/v6d, BLOCKING + linear, "
                "native KV stores and L2 loads after L1 reset",
                flush=True,
            )
        finally:
            for process in reversed(processes):
                stop(process)


if __name__ == "__main__":
    main()
