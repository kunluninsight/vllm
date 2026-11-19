#!/usr/bin/env python3
"""Summarize three Prefill monitor exports offline, with equal minute weights.

Input CSV columns match Grafana's existing exports: time/First Latency, and
Time/INPUT_TPS <service_id>/PREFIX_HIT <service_id>. First latency is milliseconds;
prefix hit remains a 0..1 ratio. Missing samples are never interpolated or zeroed.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MINUTE_MS = 60_000


def read_series(path, time_column, value_columns):
    series = {name: {} for name in value_columns}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = {time_column, *value_columns} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        for row in reader:
            timestamp = int(row[time_column])
            for name in value_columns:
                if timestamp in series[name]:
                    raise ValueError(f'{path}: duplicate timestamp {timestamp} in {name}')
                try:
                    value = float(row[name])
                except (ValueError, TypeError):
                    value = None
                series[name][timestamp] = value if value is not None and math.isfinite(value) else None
    return series


def minute_stats(series, minute, cadence_ms):
    samples = [(t, v) for t, v in series.items() if minute <= t < minute + MINUTE_MS]
    finite = [(t, v) for t, v in samples if v is not None]
    expected = set(range(minute, minute + MINUTE_MS, cadence_ms))
    valid_times = {t for t, _ in finite}
    return {
        'mean': statistics.mean(v for _, v in finite) if finite else None,
        'valid_samples': len(finite),
        'invalid_samples': len(samples) - len(finite),
        'missing_sample_times_ms': sorted(expected - valid_times),
        'off_grid_sample_times_ms': sorted(valid_times - expected),
    }


def summarize(cases, sources):
    summaries, minute_rows = [], []
    for case in cases:
        start = math.floor(float(case['start_unix_s']) / 60) * MINUTE_MS + 2 * MINUTE_MS
        end = start + 3 * MINUTE_MS
        expected = list(range(start, end, MINUTE_MS))
        within_admissions = end <= float(case['admission_stop_unix_s']) * 1000
        outcomes = case['outcomes']
        row = {
            'case': case['case'], 'concurrency': case['concurrency'],
            'from_ms': start, 'to_exclusive_ms': end, 'minutes': 3,
            'sent': case['sent'], 'real_success': outcomes['success'],
            'real_errors': case['sent'] - outcomes['success'],
            'real_outcomes': outcomes, 'stop_reason': case.get('stop_reason'),
            'window_within_admissions': within_admissions,
        }
        for metric, by_side in sources.items():
            cadence = MINUTE_MS if metric == 'deployment_first_ms' else 15_000
            bins = {side: {t: minute_stats(series, t, cadence) for t in expected}
                    for side, series in by_side.items()}
            common = [t for t in expected if all(bins[s][t]['mean'] is not None for s in ('real', 'sim'))]
            row[metric + '_minutes'] = len(common)
            row[metric + '_missing_minutes'] = {
                side: [t for t in expected if bins[side][t]['mean'] is None]
                for side in ('real', 'sim')}
            row[metric + '_missing_native_samples'] = {
                side: sum(len(bins[side][t]['missing_sample_times_ms']) for t in expected)
                for side in ('real', 'sim')}
            complete = len(common) == 3 and within_admissions
            row[metric + '_coverage'] = (
                'outside_admissions' if not within_admissions else
                'missing_minutes' if len(common) != 3 else
                'partial_native_samples' if any(row[metric + '_missing_native_samples'].values()) else 'complete')
            for side in ('real', 'sim'):
                # Do not silently shorten a case when one side lacks a minute.
                row[side + '_' + metric] = statistics.mean(bins[side][t]['mean'] for t in expected) if complete else None
            real, sim = row['real_' + metric], row['sim_' + metric]
            row[metric + '_error_pct'] = (sim / real - 1) * 100 if real not in (None, 0) and sim is not None else None
            if metric == 'PREFIX_HIT':
                row['prefix_difference_pp'] = (sim - real) * 100 if real is not None and sim is not None else None
            for timestamp in expected:
                point = {'case': case['case'], 'concurrency': case['concurrency'],
                         'metric': metric, 'time_ms': timestamp,
                         'real': bins['real'][timestamp]['mean'], 'sim': bins['sim'][timestamp]['mean'],
                         'paired': timestamp in common, 'expected_samples_per_side': MINUTE_MS // cadence}
                for side in ('real', 'sim'):
                    point.update({side + '_' + key: value for key, value in bins[side][timestamp].items() if key != 'mean'})
                minute_rows.append(point)
        summaries.append(row)
    return summaries, minute_rows


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                          if isinstance(value, (dict, list)) else value for key, value in row.items()}
                         for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('real-first', 'sim-first', 'input-prefix', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--real-service-id', required=True)
    parser.add_argument('--sim-service-id', required=True)
    args = parser.parse_args()
    if args.real_service_id == args.sim_service_id:
        raise ValueError('real and sim service IDs must differ')
    cases = json.loads(args.cases.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a nonempty client summary list')
    if len({case['case'] for case in cases}) != len(cases):
        raise ValueError('duplicate case names')
    service_ids = {'real': args.real_service_id, 'sim': args.sim_service_id}
    combined = read_series(args.input_prefix, 'Time',
                           [metric + ' ' + sid for metric in ('INPUT_TPS', 'PREFIX_HIT') for sid in service_ids.values()])
    sources = {'deployment_first_ms': {
        'real': read_series(args.real_first, 'time', ['First Latency'])['First Latency'],
        'sim': read_series(args.sim_first, 'time', ['First Latency'])['First Latency']}}
    for metric in ('INPUT_TPS', 'PREFIX_HIT'):
        sources[metric] = {side: combined[metric + ' ' + sid] for side, sid in service_ids.items()}
    summaries, minutes = summarize(cases, sources)
    for row in summaries:
        row.update(real_service_id=args.real_service_id, sim_service_id=args.sim_service_id)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    write_csv(args.output / 'summary.csv', summaries)
    write_csv(args.output / 'minute.csv', minutes)
    print(json.dumps({'cases': len(summaries), 'output': str(args.output),
                      'incomplete': [{'case': r['case'], 'metric': m, 'coverage': r[m + '_coverage']}
                                     for r in summaries for m in sources if r[m + '_coverage'] != 'complete']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
