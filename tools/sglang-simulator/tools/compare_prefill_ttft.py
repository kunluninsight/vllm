"""Compare one bounded real/sim Prefill trace window; times are process-local ms."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def window(root):
    admissions = defaultdict(list)
    scheduled, accepted, updates = defaultdict(list), defaultdict(list), {}
    external, business = defaultdict(set), defaultdict(set)
    freeze, handshake, pkg1 = defaultdict(list), defaultdict(list), defaultdict(list)
    for file in sorted(root.rglob("*.jsonl")):
        if not file.name.startswith(("events.", "lifecycle-")):
            continue
        with file.open() as source:
            for line in source:
                r = json.loads(line)
                e = r["event"]
                d = r.get("data", r)
                if not isinstance(d, dict) or d.get("native_ok") is False:
                    continue
                proc = r.get("process_id") or str(file.parent)
                rid = d.get("request_id_sha256")
                ext = d.get("external_request_id_sha256")
                bid = d.get("business_request_id_sha256")
                end, step = d.get("end_mono_ns"), d.get("step_id")
                key = proc, rid
                if e == "request.identity_alias" and rid and ext:
                    external[rid].add(ext)
                elif e == "request.business_external_alias" and bid and ext:
                    business[ext].add(bid)
                elif e in ("request.scheduler_add", "request.add") and rid and end:
                    admissions[key].append(end)
                elif e in ("scheduler.scheduled", "request.step_membership"):
                    for m in d.get("requests", d.get("members", [])):
                        t = d.get("schedule_return_mono_ns", end)
                        if t and m.get("request_id_sha256"):
                            scheduled[proc, m["request_id_sha256"]].append(t)
                elif e == "scheduler.output_update" and step is not None and end:
                    updates[proc, step] = end
                elif e in ("request.output_accepted", "request.output_observed"):
                    if rid and (d.get("new_token_count") or 0) > 0:
                        accepted[key].append((end, step))
                elif e.startswith("request.pd_") and bid and end:
                    if e == "request.pd_handshake_built":
                        handshake[proc,bid].append(end)
                    elif e == "request.pd_first_token_built":
                        pkg1[proc,bid].append(d["start_mono_ns"])
                        value = d.get("native_prefill_first_token_delay_ms")
                        if value is not None:
                            freeze[proc,bid].append(value)
    sessions = defaultdict(list)
    for key in pkg1:
        sessions[key[1]].append(key)
    rows = []
    for (proc,rid), starts in admissions.items():
        key = proc,rid
        ids = {b for x in external[rid] for b in business[x]}
        bid = next(iter(ids)) if len(ids) == len(external[rid]) == 1 else None
        row = {"business": bid, "request": rid, "core_ms": None,
               "to_first_schedule_ms": None, "schedule_to_token_ms": None,
               "pd_native_ms": None, "handshake_to_pkg1_ms": None}
        rows.append(row)
        if len(starts) != 1:  # retry/reuse: do not attach output to either admission
            row["business"] = None
            continue
        start = starts[0]
        ends = [t or updates.get((proc,s)) for t,s in accepted[key]]
        end = min((t for t in ends if t is not None and t >= start), default=None)
        first = min((t for t in scheduled[key] if t >= start), default=None)
        if end is not None:
            row["core_ms"] = (end-start)/1e6
        if first is not None and (end is None or first <= end):
            row["to_first_schedule_ms"] = (first-start)/1e6
            if end is not None:
                row["schedule_to_token_ms"] = (end-first)/1e6
        frames = sessions.get(bid, ())
        if len(frames) == 1:
            k = frames[0]
            if len(pkg1[k]) == len(freeze[k]) == 1:
                row["pd_native_ms"] = freeze[k][0]
            if len(handshake[k]) == len(pkg1[k]) == 1 and pkg1[k][0] >= handshake[k][0]:
                row["handshake_to_pkg1_ms"] = (pkg1[k][0]-handshake[k][0])/1e6
    return rows


def compare(real, sim):
    def unique(rows):
        groups = defaultdict(list)
        for r in rows:
            if r["business"]:
                groups[r["business"]].append(r)
        return {k:v[0] for k,v in groups.items() if len(v)==1}
    left,right = unique(real),unique(sim)
    pairs = [(left[k],right[k]) for k in sorted(left.keys() & right.keys())]
    result = {"admissions": [len(real),len(sim)], "unique_business_ids": [len(left),len(right)],
              "paired_business_ids": len(pairs), "phases": {}}
    for field in ("to_first_schedule_ms", "schedule_to_token_ms", "core_ms",
                  "pd_native_ms", "handshake_to_pkg1_ms"):
        realvals = [a[field] for a in real if a[field] is not None]
        simvals = [a[field] for a in sim if a[field] is not None]
        delta = [a[field]-b[field] for a,b in pairs if a[field] is not None and b[field] is not None]
        result["phases"][field] = {
            "real_n":len(realvals), "sim_n":len(simvals),
            "real_mean":mean(realvals) if realvals else None, "sim_mean":mean(simvals) if simvals else None,
            "paired_n":len(delta), "paired_mean_real_minus_sim":mean(delta) if delta else None}
    return result


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("real", type=Path)
    arg.add_argument("sim", type=Path)
    args = arg.parse_args()
    for path in (args.real,args.sim):
        if not path.is_dir():
            arg.error(f"not a directory: {path}")
    print(json.dumps(compare(window(args.real),window(args.sim)), indent=2))
