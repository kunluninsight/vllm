"""Run vLLM's CLI with an original-token dataset and timestamp replay.

Usage: python -m hisim.benchmark.serve bench serve --dataset-name hisim ...
All HTTP requests, response timing and result metrics belong to vLLM.
"""

import hashlib
import json
import math
from pathlib import Path

from hisim.benchmark.trace import load_trace, replay_requests


def install_adapter():
    from vllm.benchmarks import serve
    from vllm.benchmarks.datasets import SampleRequest

    original_args = serve.add_cli_args
    original_samples = serve.get_samples
    original_requests = serve.get_request
    state = {}

    def add_args(parser):
        original_args(parser)
        action = next(
            action for action in parser._actions if action.dest == "dataset_name"
        )
        action.choices = [*action.choices, "hisim"]
        parser.add_argument(
            "--timestamp-scale",
            type=float,
            default=1.0,
            help="Multiply relative arrival times: 0.5 is twice as fast; 2 is twice as slow.",
        )
        parser.add_argument(
            "--timestamp-unit", choices=("s", "ms", "us", "ns"), default="s"
        )
        parser.add_argument(
            "--trace-manifest",
            help="Write trace identity and send-lag diagnostics to this JSON file.",
        )

    def samples(args, tokenizer):
        if args.dataset_name != "hisim":
            return original_samples(args, tokenizer)
        if not math.isfinite(args.timestamp_scale) or args.timestamp_scale <= 0:
            raise ValueError("timestamp-scale must be finite and positive")
        if args.request_rate != float("inf") or args.ramp_up_strategy is not None:
            raise ValueError(
                "hisim arrivals use timestamps; do not combine with request-rate/ramp-up"
            )
        rows = load_trace(args.dataset_path, args.num_prompts, args.timestamp_unit)
        requests = []
        for row in rows:
            request = SampleRequest(
                prompt={"prompt_token_ids": row.input_ids},
                prompt_token_ids=row.input_ids,
                prompt_len=len(row.input_ids),
                expected_output_len=row.output_length,
                request_id=args.request_id_prefix + row.request_id,
            )
            request.hisim_timestamp = row.timestamp
            requests.append(request)
        state.update(
            scale=args.timestamp_scale, path=args.trace_manifest, requests=requests
        )
        digest = hashlib.sha256()
        for row in rows:
            digest.update(
                json.dumps(vars(row), sort_keys=True, separators=(",", ":")).encode()
            )
        state["manifest"] = {
            "dataset": str(Path(args.dataset_path).resolve()),
            "selected_rows": len(rows),
            "selected_sha256": digest.hexdigest(),
            "timestamp_unit": args.timestamp_unit,
            "timestamp_scale": args.timestamp_scale,
            "arrival_span_seconds": (rows[-1].timestamp - rows[0].timestamp)
            * args.timestamp_scale,
            "max_concurrency": args.max_concurrency,
            "input_tokens": sum(len(row.input_ids) for row in rows),
            "output_tokens": sum(row.output_length for row in rows),
            "native_metrics_unchanged": True,
        }
        print("HiSim trace:", json.dumps(state["manifest"]))
        args.metadata = [
            *(args.metadata or []),
            "arrival=trace",
            f"timestamp_scale={args.timestamp_scale}",
            f"trace_sha256={digest.hexdigest()}",
        ]
        return requests

    async def requests(input_requests, *args, **kwargs):
        if input_requests and hasattr(input_requests[0], "hisim_timestamp"):
            async for request in replay_requests(input_requests, state["scale"]):
                yield request
        else:
            async for request in original_requests(input_requests, *args, **kwargs):
                yield request

    def save_manifest():
        if state.get("path"):
            manifest = state["manifest"]
            lags = sorted(
                getattr(request, "hisim_send_lag_seconds", 0)
                for request in state["requests"]
            )
            manifest["send_lag_seconds"] = {
                "max": max(lags),
                "mean": sum(lags) / len(lags),
            }
            path = Path(state["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2))

    serve.add_cli_args = add_args
    serve.get_samples = samples
    serve.get_request = requests
    return save_manifest


def main():
    save_manifest = install_adapter()
    from vllm.entrypoints.cli.main import main as vllm_main

    try:
        vllm_main()
    finally:
        save_manifest()


if __name__ == "__main__":
    main()
