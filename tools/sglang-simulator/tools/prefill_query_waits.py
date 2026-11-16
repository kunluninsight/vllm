"""Split first-dispatch waiting using local clocks and request/generation joins."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def union_ns(spans):
    total, right = 0, None
    for a, b in sorted(spans):
        if b < a:
            raise ValueError("negative local-clock interval")
        total += max(0, b - max(a, right if right is not None else a))
        right = max(b, right if right is not None else b)
    return total


def analyze(root, receipts, side):
    business, aliases, adds, members, contexts, dispatch = {}, defaultdict(set), {}, {}, {}, {}
    queries = defaultdict(lambda: defaultdict(dict))
    capabilities = False
    for f in sorted((root / side).glob("*.jsonl")):
        for line in f.open():
            if not line.startswith("{"):
                continue
            event = json.loads(line)
            d = event.get("data", event)
            if not isinstance(d, dict):
                continue
            e, rid = event["event"], d.get("request_id_sha256")
            if e == "request.business_external_alias":
                business[d["business_request_id_sha256"]] = d["external_request_id_sha256"]
            elif e == "request.identity_alias":
                aliases[d.get("external_request_id_sha256")].add(rid)
            elif e in ("request.add", "request.scheduler_add"):
                adds[rid] = d["end_mono_ns"]
            elif e in ("request.step_membership", "scheduler.scheduled"):
                members[d.get("schedule_id", d.get("step_id"))] = d.get("members", d.get("requests"))
            elif e == "step.inputs":
                contexts[tuple(event["context"])] = d["schedule_id"]
            elif e == "engine.dispatch.observed":
                dispatch[tuple(event["context"])] = int(d["slot_dispatched"] * 1e9)
            elif e == "pp.dispatch":
                dispatch[d["step_id"]] = int(d["dispatched_at"] * 1e9)
            elif e == "request.v6d_query_capabilities":
                capabilities = capabilities or d.get("installed") is True
            elif e.startswith("request.v6d_query_") and rid:
                capabilities = True
                queries[rid][d["generation"]][e.removeprefix("request.v6d_query_")] = d
    first = {}
    for key, when in dispatch.items():
        step = contexts.get(key) if side == "real" else key
        for member in members.get(step, ()):
            rid = member["request_id_sha256"]
            first[rid] = min(first.get(rid, when), when)
    rows, detail = [], []
    for receipt in receipts:
        key = hashlib.sha256(receipt["request_id"].encode()).hexdigest()
        candidates = aliases.get(business.get(key), set()) & adds.keys() & first.keys()
        row = dict(side=side, sample=receipt["sample"], request_id=receipt["request_id"],
                   matched=len(candidates) == 1, query_hook_observed=capabilities)
        if len(candidates) != 1:
            rows.append(row)
            continue
        rid, = candidates
        admit, sent = adds[rid], first[rid]
        row["admit_to_first_dispatch_ms"] = (sent - admit) / 1e6
        spans = defaultdict(list)
        count, incomplete = 0, 0
        for generation, parts in sorted(queries[rid].items()):
            created = parts.get("created", parts.get("start", parts.get("end", {}))).get("created_mono_ns")
            if created is None or created > sent:
                continue
            count += 1
            start = parts.get("start", parts.get("end", {})).get("start_mono_ns")
            end = parts.get("end", {}).get("end_mono_ns")
            consumed = parts.get("consume", {}).get("start_mono_ns")
            incomplete += any(x is None for x in (created, start, end, consumed))
            for phase, a, b in (("query_start_wait", created, start),
                                ("query_execution", start, end),
                                ("query_ready_wait", end, consumed)):
                if a is not None and b is not None and max(admit, a) <= min(sent, b):
                    spans[phase].append((max(admit, a), min(sent, b)))
            detail.append(dict(side=side, sample=receipt["sample"], generation=generation,
                               created_from_admit_ms=(created-admit)/1e6,
                               start_from_admit_ms=(start-admit)/1e6 if start is not None else None,
                               end_from_admit_ms=(end-admit)/1e6 if end is not None else None,
                               consumed_from_admit_ms=(consumed-admit)/1e6 if consumed is not None else None,
                               hit_tokens=parts.get("end", {}).get("hit_tokens"),
                               native_ok=parts.get("end", {}).get("native_ok")))
        row.update(query_count=count if capabilities else None,
                   incomplete_queries=incomplete if capabilities else None)
        for phase in ("query_start_wait", "query_execution", "query_ready_wait"):
            row[phase+"_ms"] = union_ns(spans[phase])/1e6 if capabilities else None
        covered = union_ns([x for values in spans.values() for x in values])
        row["outside_observed_query_intervals_ms"] = (sent-admit-covered)/1e6 if capabilities and not incomplete else None
        rows.append(row)
    return rows, detail


def write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    assert union_ns([(0, 10), (5, 15), (20, 25)]) == 20
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("receipts", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    receipts = list(csv.DictReader(a.receipts.open()))
    rows, detail = [], []
    for side in ("real", "sim"):
        r, d = analyze(a.trace, receipts, side)
        rows.extend(r)
        detail.extend(d)
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / "query-waits.csv", rows)
    write(a.output / "query-generations.csv", detail)
    print(json.dumps(dict(request_rows=len(rows), matched=sum(r["matched"] for r in rows),
                          query_generations=len(detail))))
