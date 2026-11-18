# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils import RemoteOpenAIServer


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
    tmp_path, model_path, mode, backend, dataset
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
        env_dict={
            "CUDA_VISIBLE_DEVICES": "",
            "VLLM_PLUGINS": "simulator",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        },
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
