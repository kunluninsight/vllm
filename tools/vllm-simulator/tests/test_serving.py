# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import pytest
import requests

from tests.utils import RemoteOpenAIServer


@pytest.fixture
def server_env():
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "VLLM_PLUGINS": "simulator",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }


@pytest.mark.parametrize(
    "mode,backend,dataset",
    [
        ("BLOCKING", "openai", "random"),
        ("NON_BLOCKING", "openai", "random"),
        ("NON_BLOCKING", "openai", "timed_trace"),
        ("NON_BLOCKING", "openai-chat", "random"),
    ],
)
def test_bench_serve_reports_workload_timing(
    tmp_path, model_path, mode, backend, dataset, server_env
):
    model = model_path("Qwen/Qwen3-4B")
    with RemoteOpenAIServer(
        model,
        [
            "--load-format=dummy",
            "--max-model-len=256",
            "--additional-config",
            json.dumps({"simulator": {"mode": mode}}),
        ],
        env_dict=server_env,
        max_wait_seconds=120,
    ) as server:
        client = ["vllm.entrypoints.cli.main", "bench", "serve"]
        if mode == "NON_BLOCKING":
            client = [
                "vllm_simulator.benchmark.bench_serving",
                '--extra-body={"stream":false}',
            ]
        endpoint = "chat/completions" if backend == "openai-chat" else "completions"
        dataset_args = [f"--dataset-name={dataset}"]
        if dataset == "timed_trace":
            profile = tmp_path / "frontend.json"
            profile.write_text(
                '{"ingress_ms":4,"ingress_us_per_token":125,"first_chunk_ms":8,'
                '"first_chunk_us_per_token":250,"next_chunk_ms":2}'
            )
            client.append(f"--frontend-timing={profile}")
            path = tmp_path / "trace.jsonl"
            with path.open("w") as f:
                for index, ts in enumerate((0, 0.2, 2.5)):
                    row = {"timestamp": ts, "input_length": 32, "output_length": 3}
                    print(json.dumps({**row, "hash_ids": [index, index]}), file=f)
            dataset_args.append(f"--dataset-path={path}")
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                *client,
                f"--model={model}",
                f"--base-url={server.url_root}",
                f"--backend={backend}",
                f"--endpoint=/v1/{endpoint}",
                *dataset_args,
                "--input-len=32",
                "--output-len=3",
                "--num-prompts=3",
                "--request-rate=1",
                "--burstiness=inf",
                "--num-warmups=2",
                "--percentile-metrics=ttft,tpot,itl,e2el",
                "--save-result",
                f"--result-filename={tmp_path / 'bench.json'}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        result = json.loads((tmp_path / "bench.json").read_text())
        assert result["completed"] == 3 and result["failed"] == 0
        assert result["total_output_tokens"] == 9
        assert result["mean_e2el_ms"] >= 30 - 1e-9
        if mode == "NON_BLOCKING":
            calibrated = dataset == "timed_trace"
            assert result["mean_ttft_ms"] == pytest.approx(34 if calibrated else 10)
            assert result["mean_tpot_ms"] == pytest.approx(3 if calibrated else 10)
            assert result["p99_e2el_ms"] == pytest.approx(40 if calibrated else 30)
            duration = 2.54 if calibrated else 3.03
            assert result["duration"] == pytest.approx(duration)
            assert result["output_throughput"] == pytest.approx(9 / duration)
            assert result["mean_itl_ms"] == pytest.approx(3 if calibrated else 10)


@pytest.fixture
def broker_server():
    from vllm_simulator.plugin.kv_transfer.broker import BrokerServer, TransferBroker

    with BrokerServer(("127.0.0.1", 0), TransferBroker(1e9)) as broker:
        thread = threading.Thread(target=broker.serve_forever)
        thread.start()
        try:
            yield broker
        finally:
            broker.shutdown()
            thread.join()


@pytest.mark.parametrize("connector", ["NixlConnector", "MooncakeConnector"])
def test_native_connector_config_serves_pd_handoff(
    model_path, server_env, broker_server, connector
):
    from vllm_simulator.plugin.kv_transfer.broker import TransferBroker

    model = model_path("Qwen/Qwen3-32B-FP8")
    broker_url = f"http://127.0.0.1:{broker_server.server_port}"
    with ExitStack() as servers:
        endpoints = []
        for role, tp in [("kv_producer", 1), ("kv_consumer", 2)]:
            simulator = {
                "mode": "BLOCKING",
                "kv_transfer": {"broker_url": broker_url, "kv_role": role},
            }
            server = servers.enter_context(
                RemoteOpenAIServer(
                    model,
                    [
                        "--load-format=dummy",
                        "--max-model-len=64",
                        "--block-size=16",
                        "--num-gpu-blocks-override=5",
                        f"--tensor-parallel-size={tp}",
                        "--kv-transfer-config",
                        json.dumps({"kv_connector": connector, "kv_role": "kv_both"}),
                        "--additional-config",
                        json.dumps({"simulator": simulator}),
                    ],
                    env_dict=server_env,
                    max_wait_seconds=120,
                )
            )
            endpoints.append(server.url_for("v1", "completions"))

        def submit(url, tokens, handoff, first_token):
            return requests.post(
                url,
                json={
                    "model": model,
                    "prompt": list(range(first_token, first_token + 33)),
                    "max_tokens": tokens,
                    "ignore_eos": True,
                    "kv_transfer_params": handoff,
                },
                timeout=30,
            )

        def publish(first_token):
            response = submit(endpoints[0], 1, {"do_remote_decode": True}, first_token)
            response.raise_for_status()
            result = response.json()
            assert result["usage"]["completion_tokens"] == 1
            return result["kv_transfer_params"]

        def decode(first_token):
            handoff = publish(first_token)
            del handoff["transfer_id"]  # Routers may forward only the native IDs.
            response = submit(endpoints[1], 2, handoff, first_token)
            response.raise_for_status()
            result = response.json()
            assert result["usage"]["completion_tokens"] == 2
            return result["kv_transfer_params"]

        result = decode(100)
        assert result["simulation"]["ext_kv_hit_len"] == 33
        timing = result["simulation_kv_transfer"]
        assert 0 <= timing["kv_destination_wait_s"] <= timing["kv_offer_to_claim_s"]
        old = broker_server.broker
        old.latency = 60
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(submit, endpoints[1], 2, publish(200), 200)
            deadline = time.monotonic() + 15
            while not old.stats()["waiting_receivers"]:
                assert time.monotonic() < deadline, "D did not claim the offer"
                time.sleep(0.01)
            broker_server.broker = TransferBroker(1e9)
            assert pending.result(timeout=15).status_code >= 400
        # A fresh three-block prompt only fits once both old pools drain.
        assert decode(300)["simulation"]["output_length"] == 2
