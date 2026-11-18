"""Finite concurrency or open-loop request-rate Prefill comparison."""
import argparse
import asyncio
import copy
import importlib.util
import json
import math
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


def _ramp_up_seconds(args, request_rate):
    seconds = getattr(args, "ramp_up_seconds", 0.0)
    if not 0 <= seconds < args.duration:
        raise ValueError("--ramp-up-seconds must satisfy 0 <= ramp < duration")
    if request_rate is not None and seconds:
        raise ValueError("--ramp-up-seconds is supported only in closed-loop concurrency mode")
    return seconds


async def case(args, sender, target, template, key, concurrency, case_index, request_rate=None):
    ramp_up_seconds = _ramp_up_seconds(args, request_rate)
    name = f"c{concurrency}" if request_rate is None else "r" + format(request_rate, "g").replace(".", "p")
    out = args.output / name
    out.mkdir(mode=0o700, exist_ok=False)
    run_id = str(uuid.uuid4())
    namespace = f"{args.label}-{name}"
    mono0, wall0 = time.monotonic(), time.time()
    deadline = mono0 + args.duration
    records, issued, abort = {}, 0, False
    stop_reason, peak_inflight = None, 0
    manifest = dict(run_id=run_id, label=namespace, concurrency=concurrency,
                    arrival_mode="closed_loop" if request_rate is None else "constant_rate",
                    request_rate=request_rate, ramp_up_seconds=ramp_up_seconds,
                    inflight_limit_behavior="stop admissions, then drain; never throttle rate",
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
            async def submit(index, at_s):
                nonlocal abort, stop_reason
                body, family = payload(template, manifest["seed"], namespace, index)
                encoded = sender.encode({**body, "model": target["model"], "stream": True,
                                         "stream_options": {"include_usage": True}})
                item = dict(id=f"{namespace}-{index:06d}", at_s=at_s,
                            input_line=index+1, payload_sha256=sender.sha(encoded), body=encoded)
                await sender.send(item, session, sem, target, run_id, mono0, wall0, log, records)
                rec = records[item["id"]]
                if rec["http_status"] in (400, 401, 403, 404):
                    abort, stop_reason = True, "auth_or_route"
                if len(records) % 100 == 0:
                    print(json.dumps(dict(case=namespace, completed=len(records),
                                          elapsed_s=round(time.monotonic()-mono0, 1))), flush=True)

            async def worker(worker_index):
                nonlocal issued
                if ramp_up_seconds:
                    await asyncio.sleep(max(0, mono0 + ramp_up_seconds * worker_index / concurrency
                                            - time.monotonic()))
                while not abort and time.monotonic() < deadline and issued < args.max_requests:
                    index = issued
                    issued += 1
                    await submit(index, time.monotonic()-mono0)

            if request_rate is None:
                await asyncio.gather(*(worker(i) for i in range(concurrency)))
            else:
                active = set()
                while issued < args.max_requests:
                    # Arrival times depend on the clock, never response completion.
                    at_s = issued / request_rate
                    if at_s >= args.duration:
                        break
                    await asyncio.sleep(max(0, mono0 + at_s - time.monotonic()))
                    if abort:
                        break
                    if time.monotonic() >= deadline:
                        stop_reason = "client_late_at_deadline"
                        break
                    done = {task for task in active if task.done()}
                    for task in done:
                        task.result()
                    active -= done
                    if len(active) >= concurrency:
                        stop_reason = "inflight_limit"
                        break
                    if time.monotonic() - mono0 - at_s > args.max_send_lag:
                        stop_reason = "client_send_lag"
                        break
                    active.add(asyncio.create_task(submit(issued, at_s)))
                    issued += 1
                    peak_inflight = max(peak_inflight, len(active))
                await asyncio.gather(*active)
    ended = time.time()
    counts = {s: sum(r["outcome"] == s for r in records.values())
              for s in ("success", "error", "timeout", "cancelled", "not_sent")}
    delays = sorted(r["send_delay_s"] for r in records.values() if r.get("send_delay_s") is not None)
    actual_sends = [r["actual_send_unix_s"] for r in records.values() if r.get("actual_send_unix_s") is not None]
    summary = dict(case=namespace, concurrency=concurrency, sent=issued,
                   arrival_mode=manifest["arrival_mode"], request_rate=request_rate,
                   ramp_up_seconds=ramp_up_seconds,
                   stop_reason=stop_reason, peak_client_inflight=peak_inflight if request_rate is not None else None,
                   scheduled_requests=min(args.max_requests, math.ceil(args.duration * request_rate)) if request_rate is not None else None,
                   actual_send_count=len(actual_sends),
                   send_lag_p95_s=delays[int(.95*(len(delays)-1))] if delays else None,
                   send_lag_max_s=max(delays) if delays else None,
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
    levels = args.request_rate if args.request_rate is not None else args.concurrency
    for i, level in enumerate(levels):
        rate = level if args.request_rate is not None else None
        c = args.max_inflight if rate is not None else level
        summary = await case(args, sender, target, template, key, c, i, rate)
        summaries.append(summary)
        (args.output / "cases.json").write_text(json.dumps(summaries, indent=2))
        if summary["stop_reason"] or not summary["all_success"]:
            break
        if i + 1 < len(levels):
            print(json.dumps(dict(gap_seconds=args.gap, after_case=level, start_unix_s=time.time())), flush=True)
            await asyncio.sleep(args.gap)
    return 0 if all(x["all_success"] and not x["stop_reason"] for x in summaries) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sender", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    modes.add_argument("--request-rate", type=float, nargs="+")
    p.add_argument("--max-inflight", type=int, default=1024)
    p.add_argument("--max-send-lag", type=float, default=0.25)
    p.add_argument("--duration", type=float, default=600)
    p.add_argument("--ramp-up-seconds", type=float, default=0,
                   help="spread closed-loop worker starts over this interval; excluded from request latency")
    p.add_argument("--gap", type=float, default=300)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--max-requests", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1161637)
    a = p.parse_args()
    try:
        _ramp_up_seconds(a, a.request_rate)
    except ValueError as error:
        p.error(str(error))
    assert 1 <= a.max_inflight <= 2048 and 0 < a.max_send_lag <= 1
    assert a.request_rate is None or (all(math.isfinite(r) and 0 < r <= 100 for r in a.request_rate) and len(set(a.request_rate)) == len(a.request_rate))
    assert 0 < a.timeout <= 300
    assert all(0 < c <= 2048 for c in a.concurrency) and len(set(a.concurrency)) == len(a.concurrency)
    assert 0 < a.duration <= 600 and 0 <= a.gap <= 300 and 0 < a.max_requests <= 20000
    raise SystemExit(asyncio.run(run(a)))
