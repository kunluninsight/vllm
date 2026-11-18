# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
import time

import pytest
import requests

from vllm.utils.network_utils import get_open_port


def test_blocking_server_returns_predicted_tokens_and_timing(tmp_path, model_path):
    model = model_path("Qwen/Qwen3-4B")
    port = get_open_port()
    url = f"http://127.0.0.1:{port}"
    log = tmp_path / "serve.log"
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "256",
        "--additional-config",
        json.dumps({"simulator": {"mode": "BLOCKING"}}),
    ]
    with (
        log.open("w") as output,
        subprocess.Popen(
            args,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "VLLM_PLUGINS": "simulator"},
            stdout=output,
            stderr=subprocess.STDOUT,
        ) as proc,
    ):
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail(log.read_text())
            started = time.monotonic()
            response = requests.post(
                url + "/v1/completions",
                json={
                    "model": model,
                    "prompt": [42, 43],
                    "max_tokens": 3,
                    "temperature": 0,
                    "ignore_eos": True,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            assert result["usage"]["completion_tokens"] == 3
            assert result["kv_transfer_params"]["simulation"]["output_length"] == 3
            assert time.monotonic() - started >= 0.03
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
