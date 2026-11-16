"""Finite closed-loop Prefill comparison; reuse the validated SSE sender."""
import argparse
import asyncio
import copy
import importlib.util
import json
import os
import random
import time
import uuid
from pathlib import Path

WORDS = "river cedar bridge copper garden sugar velvet pencil radio tiger autumn willow basket comet lemon orbit chair rabbit paper coral".split()
LENGTHS = (1250, 3200, 4300, 3600)


def payload(template, seed, namespace, index):
    # Every full prompt differs; pairs share a prefix to exercise prefix caching.
    family = (index // 64) * 32 + index % 32
    words = random.Random(seed + family).choices(WORDS, k=LENGTHS[family % 4])
    body = copy.deepcopy(template)
    body.pop("max_tokens", None)
    body["max_completion_tokens"] = 4
    body["messages"][0]["content"] = (
        f"Trial {namespace} family {family}. " + " ".join(words)
        + f". Unique question {index}: Reply OK."
    )
    return body, family


async def case(args, sender, target, template, key, concurrency, case_index):
    out = args.output / f"c{concurrency}"
    out.mkdir(mode=0o700, exist_ok=False)
    run_id = str(uuid.uuid4())
    namespace = f"{args.label}-c{concurrency}"
    mono0, wall0 = time.monotonic(), time.time()
    deadline = mono0 + args.duration
    records, issued, abort = {}, 0, False
    manifest = dict(run_id=run_id, label=namespace, concurrency=concurrency,
                    duration_s=args.duration, max_requests=args.max_requests,
                    start_unix_s=wall0, start_monotonic_s=mono0,
                    seed=args.seed + case_index * 100000,
                    word_lengths=LENGTHS, full_prompt_repeats=False,
                    shared_prefix_rule="32 families per 64 requests; unique suffix",
                    next_case_gap_s=args.gap,
                    concurrency_scope="client HTTP requests through SSE completion",
                    actual_send_definition="after concurrency gate, before session.post",
                    client_ttft_definition="first nonempty content SSE; diagnostic only")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sem = asyncio.Semaphore(concurrency)
    timeout = sender.aiohttp.ClientTimeout(total=args.timeout)
    with (out / "requests.jsonl").open("x") as log:
        async with sender.aiohttp.ClientSession(
            timeout=timeout, connector=sender.aiohttp.TCPConnector(limit=concurrency),
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json", "Accept": "text/event-stream"},
            trust_env=False,
        ) as session:
            async def worker():
                nonlocal issued, abort
                while not abort and time.monotonic() < deadline and issued < args.max_requests:
                    index = issued
                    issued += 1
                    body, family = payload(template, manifest["seed"], namespace, index)
                    encoded = sender.encode({**body, "model": target["model"], "stream": True,
                                             "stream_options": {"include_usage": True}})
                    item = dict(id=f"{namespace}-{index:06d}", at_s=time.monotonic()-mono0,
                                input_line=index+1, payload_sha256=sender.sha(encoded), body=encoded)
                    await sender.send(item, session, sem, target, run_id, mono0, wall0, log, records)
                    rec = records[item["id"]]
                    if rec["http_status"] in (400, 401, 403, 404):
                        abort = True
                    if len(records) % 100 == 0:
                        print(json.dumps(dict(case=namespace, completed=len(records),
                                              elapsed_s=round(time.monotonic()-mono0, 1))), flush=True)
            await asyncio.gather(*(worker() for _ in range(concurrency)))
    ended = time.time()
    counts = {s: sum(r["outcome"] == s for r in records.values())
              for s in ("success", "error", "timeout", "cancelled", "not_sent")}
    summary = dict(case=namespace, concurrency=concurrency, sent=issued,
                   start_unix_s=wall0, admission_stop_unix_s=wall0+args.duration,
                   last_send_unix_s=max((r["actual_send_unix_s"] or wall0 for r in records.values()), default=wall0),
                   completed_unix_s=ended, elapsed_s=ended-wall0, outcomes=counts,
                   all_success=counts["success"] == issued, stopped_for_auth_or_route=abort,
                   request_limit_reached=issued == args.max_requests)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


async def run(args):
    spec = importlib.util.spec_from_file_location("verified_sender", args.sender)
    sender = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sender)
    target, _ = sender.load_target(args.target, False)
    template = json.loads(args.template.read_text().splitlines()[0])["payload"]
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("DASHSCOPE_API_KEY is not set")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    summaries = []
    for i, c in enumerate(args.concurrency):
        summary = await case(args, sender, target, template, key, c, i)
        summaries.append(summary)
        (args.output / "cases.json").write_text(json.dumps(summaries, indent=2))
        if summary["stopped_for_auth_or_route"]:
            break
        if i + 1 < len(args.concurrency):
            print(json.dumps(dict(gap_seconds=args.gap, after_case=c, start_unix_s=time.time())), flush=True)
            await asyncio.sleep(args.gap)
    return 0 if all(x["all_success"] for x in summaries) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sender", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--duration", type=float, default=600)
    p.add_argument("--gap", type=float, default=300)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--max-requests", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1161637)
    a = p.parse_args()
    assert all(0 < c <= 64 for c in a.concurrency) and len(set(a.concurrency)) == len(a.concurrency)
    assert 0 < a.duration <= 600 and 0 <= a.gap <= 300 and 0 < a.max_requests <= 20000
    raise SystemExit(asyncio.run(run(a)))
