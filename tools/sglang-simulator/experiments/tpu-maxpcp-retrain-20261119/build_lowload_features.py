"""Build exact low-load stage features from archived native scheduler events."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sglang_simulator.simulation.vllm.tpu_pp.stage_model import (
    FEATURE_NAMES, TAIL_FEATURES, PcpLayout, StageSpan, stage_features, tail_features,
)


def reuse_cases(root, sources):
    """Merge completed collect_case.py exports without touching online services."""
    events, metas = {}, []
    for source in sources:
        with Path(source["metadata_jsons"]).open() as f:
            found = [m for m in map(json.loads, f) if m["pid"] == 1644]
        assert len(found) == 1, ("core metadata identity", source["case"], len(found))
        meta = found[0]
        metas.append(meta)
        offset = meta["wall_anchor_ns"] - meta["mono_anchor_ns"]
        with Path(source["core_jsonl"]).open() as f:
            for line in f:
                e = json.loads(line)
                if e["event"] not in ("request.step_membership", "step.inputs"):
                    continue
                if not source["start"] - 1 <= (e["mono_ns"] + offset) / 1e9 <= source["end"] + 1:
                    continue
                old = events.setdefault(e["seq"], e)
                assert old == e, ("conflicting source sequence", e["seq"])
    assert len({(m["process_id"], m["wall_anchor_ns"], m["mono_anchor_ns"]) for m in metas}) == 1
    assert events
    root.mkdir(parents=True, exist_ok=True)
    (root / "windows.json").write_text(json.dumps([
        {k: s[k] for k in ("case", "concurrency", "start", "end")} for s in sources]))
    (root / "core-metadata.json").write_text(json.dumps(max(metas, key=lambda m: m["written"])))
    (root / "core-events.jsonl").write_text("".join(json.dumps(events[s], separators=(",", ":")) + "\n" for s in sorted(events)))


def build(root):
    windows = json.loads((root / "windows.json").read_text())
    meta = json.loads((root / "core-metadata.json").read_text())
    offset = meta["wall_anchor_ns"] - meta["mono_anchor_ns"]
    with (root / "core-events.jsonl").open() as source:
        events = [json.loads(line) for line in source]
    assert all(a["seq"] < b["seq"] for a, b in zip(events, events[1:]))
    memberships, inputs = {}, {}
    for e in events:
        d = e["data"]
        sid = d["schedule_id"]
        target = memberships if e["event"] == "request.step_membership" else inputs
        assert e["event"] in ("request.step_membership", "step.inputs")
        assert sid not in target, ("duplicate scheduler event", sid, e["event"])
        target[sid] = e
    layout = PcpLayout(8, 128, (256, 512, 1024))
    cases, features, selected = [], [], set()
    for w in windows:
        case = dict(case=w["case"], concurrency=w["concurrency"], start=w["start"],
                    end=w["end"], steps=[])
        for sid, e in sorted(inputs.items()):
            wall = (e["mono_ns"] + offset) / 1e9
            if not w["start"] <= wall <= w["end"]:
                continue
            d = e["data"]
            if not d["total_num_scheduled_tokens"]:
                continue
            context = e["context"]
            assert context and context[2] == sid and context[3] == "real"
            assert sid not in selected, ("overlapping case windows", sid)
            assert d["membership_complete"] and d["native_dispatch_ok"]
            assert d["query_sum_valid"] and d["past_complete"]
            members = d["members"]
            assert members and all(m["membership_complete"] for m in members)
            assert all(not m["duplicate_id"] for m in members)
            assert sid in memberships, ("missing schedule membership", sid)
            md = memberships[sid]["data"]
            assert md["native_schedule_ok"] and md["membership_complete"]
            assert md["members"] == members, ("membership differs", sid)
            assert sum(m["query_length"] for m in members) == d["total_num_scheduled_tokens"]
            spans = [StageSpan(m["query_length"], m["past_kv_length"],
                               m["input_length"], False) for m in members]
            f = {**stage_features(spans, layout), **tail_features(spans)}
            features.append(dict(case=w["case"], concurrency=w["concurrency"], step=sid, **f))
            case["steps"].append(dict(step=sid, context=context, mono_ns=e["mono_ns"],
                                      wall_s=wall, members=members))
            selected.add(sid)
        assert case["steps"], ("empty case", w["case"])
        assert len({tuple(s["context"][:2]) for s in case["steps"]}) == 1
        cases.append(case)
    with (root / "step-features.csv").open("w") as f:
        writer = csv.DictWriter(f, ["case", "concurrency", "step", *FEATURE_NAMES, *TAIL_FEATURES])
        writer.writeheader()
        writer.writerows(features)
    (root / "cases.json").write_text(json.dumps(cases, indent=2))
    plan = [dict(case=c["case"], concurrency=c["concurrency"], start=c["start"], end=c["end"],
                 lo=c["steps"][0]["step"], hi=c["steps"][-1]["step"],
                 context=c["steps"][0]["context"], steps=len(c["steps"])) for c in cases]
    (root / "native-plan.json").write_text(json.dumps(plan, indent=2))
    summary = dict(steps=len(features), cases={c["case"]: len(c["steps"]) for c in cases},
                   padding_counts=dict(Counter(f["global_padded_tokens"] for f in features)),
                   logprobs=False, scope="nonempty dispatches inside isolated loadgen case windows")
    (root / "feature-coverage.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--reuse-cases", type=Path)
    args = p.parse_args()
    if args.reuse_cases:
        reuse_cases(args.root, json.loads(args.reuse_cases.read_text()))
    print(json.dumps(build(args.root)), flush=True)
