"""Compare same-window handoff observations; no cross-host clock subtraction."""
import collections
import csv
import json
import math
from pathlib import Path

ROOT = Path('/data2/maruiyan.mry/qwen38-tpu-pp/work/pressure-1116/ramp-handoffs-1118')
CASES = json.loads((ROOT / 'case-windows.json').read_text())


def stats(values):
    v = sorted(values)
    if not v:
        return dict(n=0)
    p = (len(v) - 1) * .95
    a, b = math.floor(p), math.ceil(p)
    return dict(n=len(v), mean=sum(v) / len(v), median=(v[(len(v)-1)//2] + v[len(v)//2])/2,
                p95=v[a] + (v[b]-v[a]) * (p-a), min=v[0], max=v[-1])


def case_for(wall_s):
    matches = [c['concurrency'] for c in CASES if c['start'] - 3 <= wall_s <= c['end'] + 3]
    assert len(matches) <= 1
    return matches[0] if matches else None


assert stats([1, 2, 3])['mean'] == 2
assert stats([1, 2, 3])['p95'] == 2.9
rows, unresolved, seen_real = [], [], set()
for stage in range(4):
    meta = json.loads((ROOT / ('real%d.metadata.json' % stage)).read_text())
    offset = meta['wall_anchor_ns'] - meta['mono_anchor_ns']
    groups = {}
    for line in (ROOT / ('real%d.jsonl' % stage)).open():
        e = json.loads(line)
        assert e['rank']['pp'] == stage and e['rank']['pcp'] == 0
        assert (stage, e['seq']) not in seen_real
        seen_real.add((stage, e['seq']))
        d = e['data']
        key = (tuple(d['protocol_session']), d['burst_id'], d['launch_ordinal'])
        group = groups.setdefault(key, {})
        assert e['event'] not in group, (stage, key, e['event'])
        group[e['event']] = e
    for key, group in groups.items():
        a = group.get('native.handoff_barrier_call')
        b = group.get('native.handoff_submitted')
        if a is None or b is None:
            unresolved.append(dict(stage=stage, key=key, events=list(group)))
            continue
        d = a['data']
        n = case_for((a['mono_ns'] + offset) / 1e9)
        if n is None:
            continue
        assert d['native_ok'] is True
        assert d['exit_mono_ns'] >= d['enter_mono_ns']
        assert b['data']['observed_mono_ns'] >= d['exit_mono_ns']
        rows.append(dict(side='real', concurrency=n, stage=stage, burst=key[1], ordinal=key[2],
                         wall_s=(a['mono_ns'] + offset) / 1e9,
                         outgoing_step_id=a['context'][2] if a.get('context') else None,
                         barrier_ms=(d['exit_mono_ns']-d['enter_mono_ns'])/1e6,
                         barrier_return_to_submit_ms=(b['data']['observed_mono_ns']-d['exit_mono_ns'])/1e6,
                         native_barrier_ms=None, added_overhead_ms=None, payload_bytes=d['payload_bytes']))

seen_sim = set()
for case in CASES:
    n = case['concurrency']
    for line in (ROOT / ('sim%d.jsonl' % n)).open():
        e = json.loads(line)
        d = e.get('stage_event') or e.get('data', {}).get('stage_event')
        assert e['event'] == 'pp.stage_event' and d['kind'] == 'handoff_barrier'
        key = (n, d['stage'], d['burst'], d['transport_index'])
        assert key not in seen_sim
        seen_sim.add(key)
        assert d['exit_at'] >= d['native_return_at'] >= d['enter_at']
        assert case_for(e['time_ns']/1e9) == n
        rows.append(dict(side='sim', concurrency=n, stage=d['stage'], burst=d['burst'], ordinal=d['transport_index'],
                         wall_s=e['time_ns']/1e9, outgoing_step_id=d['outgoing_step_id'],
                         barrier_ms=(d['exit_at']-d['enter_at'])*1e3,
                         barrier_return_to_submit_ms=(d['time']-d['exit_at'])*1e3,
                         native_barrier_ms=(d['native_return_at']-d['enter_at'])*1e3,
                         added_overhead_ms=(d['exit_at']-d['native_return_at'])*1e3,
                         payload_bytes=None))

summary, group_summary, long_stalls = [], [], []
for n in (64, 128, 256, 512):
    for side in ('real', 'sim'):
        groups = collections.defaultdict(dict)
        for r in rows:
            if r['concurrency'] == n and r['side'] == side:
                key = (r['burst'], r['ordinal'])
                assert r['stage'] not in groups[key]
                groups[key][r['stage']] = r
        complete = {k: v for k, v in groups.items() if set(v) == {0, 1, 2, 3}}
        assert complete, (n, side)
        minimums = [min(r['barrier_ms'] for r in g.values()) for g in complete.values()]
        for key, g in complete.items():
            shortest = min(r['barrier_ms'] for r in g.values())
            if shortest > 100:
                long_stalls.append(dict(concurrency=n, side=side, burst=key[0], ordinal=key[1],
                                        minimum_barrier_ms=shortest, p0_outgoing_step=g[0]['outgoing_step_id'],
                                        p0_wall_s=g[0]['wall_s'], since_case_start_s=g[0]['wall_s']-next(c['start'] for c in CASES if c['concurrency']==n),
                                        per_stage_barrier_ms={s:r['barrier_ms'] for s,r in g.items()}))
        # The min of four per-process durations is observable without aligning
        # host clocks; it is NOT the intrinsic cost of a 32-rank world barrier.
        group_summary.append(dict(concurrency=n, side=side, complete_groups=len(complete), incomplete_groups=len(groups)-len(complete),
                                  min_of_four_barrier_ms=stats(minimums),
                                  min_of_four_gt100ms_count=sum(v>100 for v in minimums),
                                  min_of_four_gt100ms_sum_ms=sum(v for v in minimums if v>100),
                                  min_of_four_le100ms_conditional=stats([v for v in minimums if v<=100]),
                                  max_of_four_barrier_ms=stats([max(r['barrier_ms'] for r in g.values()) for g in complete.values()])))
        for stage in range(4):
            selected = [g[stage] for g in complete.values()]
            summary.append(dict(concurrency=n, side=side, stage=stage,
                                barrier_ms=stats([r['barrier_ms'] for r in selected]),
                                barrier_return_to_submit_ms=stats([r['barrier_return_to_submit_ms'] for r in selected]),
                                native_barrier_ms=stats([r['native_barrier_ms'] for r in selected if r['native_barrier_ms'] is not None]),
                                added_overhead_ms=stats([r['added_overhead_ms'] for r in selected if r['added_overhead_ms'] is not None]),
                                payload_bytes_counts=dict(collections.Counter(r['payload_bytes'] for r in selected))))

with (ROOT / 'handoff-observations.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
output = dict(scope='same-window online cohorts; not matched batches or isolated ICI timing', per_stage=summary,
              per_group=group_summary, long_stalls=long_stalls, unpaired_real_events=unresolved,
              real_timestamp_rule='durations use timestamps within each process; wall anchors only select case windows')
(ROOT / 'handoff-summary.json').write_text(json.dumps(output, indent=2))
for n in (64, 128, 256, 512):
    print('CASE', n)
    for stage in range(4):
        a, b = [next(r for r in summary if r['concurrency'] == n and r['stage'] == stage and r['side'] == side) for side in ('real', 'sim')]
        print(stage, 'n', a['barrier_ms']['n'], b['barrier_ms']['n'], 'barrier real/sim', a['barrier_ms']['mean'], b['barrier_ms']['mean'],
              'submit real/sim', a['barrier_return_to_submit_ms']['mean'], b['barrier_return_to_submit_ms']['mean'])
print('HANDOFF_ANALYSIS_COMPLETE', len(rows), 'unpaired_real', len(unresolved))
