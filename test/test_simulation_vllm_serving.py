"""
Test vLLM server-mode simulation with BLOCKING mode.

Follows the pattern of test_simulation_sglang_serving.py:
1. Spawns vLLM simulation server (launch_server.py) as subprocess
2. Sends requests via OpenAI-compatible /v1/completions API
3. Validates that requests complete and timing is reasonable
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from hisim.env import Envs

Envs.set_config_path(os.path.dirname(__file__) + "/assets/config_vllm.json")
Envs.set_output_mode("BLOCKING")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VLLM_FORCE_DETOKENIZE"] = "1"

MODEL_PATH = "Qwen/Qwen3-0.6B"
SERVER_PORT = 18100


class VLLMServingRunner:
    def __init__(self, model: str, port: int = SERVER_PORT, **extra_args):
        cmd = [
            sys.executable,
            "-m",
            "hisim.simulation.hijack.vllm.launch_server",
            "--model",
            model,
            "--port",
            str(port),
            "--dtype",
            "float16",
            "--load-format",
            "dummy",
            "--enforce-eager",
            "--block-size",
            "16",
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "8192",
            "--num-gpu-blocks-override",
            "1000000",
        ]
        for k, v in extra_args.items():
            flag = "--" + k.replace("_", "-")
            if v is True:
                cmd.append(flag)
            elif v is False:
                pass
            else:
                cmd.extend([flag, str(v)])

        env = os.environ.copy()
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.server_proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)

        # Wait for server to be ready
        dur = 0
        while dur < 120:
            try:
                r = requests.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
            dur += 1
        raise RuntimeError("Failed to start vLLM simulation server.")

    def completions(self, prompt: str, max_tokens: int = 10) -> dict:
        """Send a single completion request."""
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "ignore_eos": True,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown(self):
        if not self.server_proc or self.server_proc.poll() is not None:
            return
        os.killpg(self.server_proc.pid, signal.SIGTERM)
        try:
            self.server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.server_proc.pid, signal.SIGKILL)
            self.server_proc.wait()
        self.server_proc = None


def test_vllm_serving_blocking():
    """Test vLLM server with BLOCKING mode simulation."""
    runner = VLLMServingRunner(model=MODEL_PATH)

    try:
        # Send a few requests sequentially
        results = []
        for i in range(3):
            t0 = time.time()
            result = runner.completions(prompt=f"Hello world {i}", max_tokens=5)
            elapsed = time.time() - t0
            results.append((result, elapsed))

        # Validate responses
        for i, (result, elapsed) in enumerate(results):
            assert "choices" in result, f"Request {i}: no choices in response"
            assert len(result["choices"]) > 0, f"Request {i}: empty choices"
            text = result["choices"][0]["text"]
            assert len(text) > 0, f"Request {i}: empty text output"
            # In BLOCKING mode, each request should take some time due to time.sleep
            # With AIConfigurator predictor, each token takes real inference time
            print(
                f"  Request {i}: output_tokens={result['usage']['completion_tokens']}, "
                f"elapsed={elapsed:.3f}s"
            )

        print("\n[PASS] test_vllm_serving_blocking passed!")
    finally:
        runner.shutdown()


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_vllm_serving_blocking_dp2(tmp_path=None):
    """Internal-LB DP2: both ranks serve and agree on every coordinated batch."""
    output_dir = str(tmp_path or tempfile.mkdtemp(prefix="hisim_dp2_"))
    saved = {k: os.environ.get(k) for k in ("HISIM_OUTPUT_DIR", "HISIM_RECORD_STATS")}
    os.environ["HISIM_OUTPUT_DIR"] = output_dir
    os.environ["HISIM_RECORD_STATS"] = "1"
    try:
        runner = VLLMServingRunner(
            model=MODEL_PATH,
            port=SERVER_PORT + 1,
            data_parallel_size=2,
            profiler_config=json.dumps({"profiler": "cuda"}),
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    try:
        num_requests, max_tokens = 16, 8
        prompts = [" ".join(["token"] * (64 + 32 * i)) for i in range(num_requests)]
        with ThreadPoolExecutor(max_workers=num_requests) as pool:
            results = list(
                pool.map(
                    lambda p: runner.completions(p, max_tokens=max_tokens), prompts
                )
            )
        for i, result in enumerate(results):
            assert result["usage"]["completion_tokens"] == max_tokens, i
        # A lone request leaves the other rank running coordinated dummies.
        solo = runner.completions(" ".join(["solo"] * 100), max_tokens=16)
        assert solo["usage"]["completion_tokens"] == 16
        num_requests += 1

        # Profile fans out to both engines; each exports into its dp{rank} dir.
        requests.post(f"{runner.base_url}/start_profile", timeout=60).raise_for_status()
        reqs = {r: _load_jsonl(f"{output_dir}/dp{r}/request.jsonl") for r in (0, 1)}
        iters = {r: _load_jsonl(f"{output_dir}/dp{r}/iteration.jsonl") for r in (0, 1)}
        assert len(reqs[0]) + len(reqs[1]) == num_requests, reqs
        # Internal LB must route requests to both ranks.
        assert reqs[0] and reqs[1], {r: len(v) for r, v in reqs.items()}

        by_step = {}
        for rank, rows in iters.items():
            assert rows, f"dp{rank} recorded no iterations"
            for row in rows:
                assert (row["dp_rank"], row["dp_size"]) == (rank, 2)
                assert row["requests_scope"] == "dp"
                by_dp = row["requests_by_dp"]
                # Only real local steps are recorded; peers may run dummies.
                assert by_dp[rank], row
                assert row["requests"] == [r for reqs_ in by_dp for r in reqs_]
                by_step.setdefault(row["dp_step_id"], {})[rank] = row
        joint = [s for s in by_step.values() if len(s) == 2]
        assert joint, "no step had real batches on both DP ranks"
        for rows in joint:
            assert rows[0]["requests_by_dp"] == rows[1]["requests_by_dp"]
            assert rows[0]["full_step_latency"] == rows[1]["full_step_latency"]
        # Steps recorded on one rank only paired a real batch with a dummy.
        solo_steps = [s for s in by_step.values() if len(s) == 1]
        assert solo_steps, "no step paired a real batch with a dummy batch"
        for rows in solo_steps:
            ((rank, row),) = rows.items()
            assert row["requests_by_dp"][1 - rank] == [], row
        print(
            f"\n[PASS] test_vllm_serving_blocking_dp2 passed! requests/rank="
            f"{len(reqs[0])}/{len(reqs[1])}, iterations/rank="
            f"{len(iters[0])}/{len(iters[1])}, joint steps={len(joint)}, "
            f"real+dummy steps={len(solo_steps)}"
        )
    finally:
        runner.shutdown()


if __name__ == "__main__":
    test_vllm_serving_blocking()
    test_vllm_serving_blocking_dp2()
